import os
import sys

import time
import itertools
import numpy as np
import torch
from matplotlib.pylab import plt

from libs.dataloader.View import View, ViewCreator
from libs.dataloader.Scene import Scene

lib_decompre = "/home/dotamthuc-3090/Projects/ViewSynthesis/Differentiable3DPC-Decompression-and-Rendering"
sys.path.insert(0, lib_decompre)
import decompress_render


lib_svraster = "/home/dotamthuc-3090/Projects/ViewSynthesis/svraster"
sys.path.insert(0, lib_svraster)
from src.sparse_voxel_model import SparseVoxelModel


# torch.set_printoptions(
#     precision=4,
#     sci_mode=False,
#     threshold=10000,    # Print the full array if it has fewer than 10000 elements
#     linewidth=1000       # Number of characters per line
# )
# np.set_printoptions(
#     precision=4,        # Number of digits of precision for floating point output
#     suppress=False,      # Suppress scientific notation
#     threshold=10000,    # Print the full array if it has fewer than 10000 elements
#     linewidth=1000       # Number of characters per line
# )


def resize_rendering(render, size, mode='bilinear', align_corners=False):
    return torch.nn.functional.interpolate(
        render[None], size=size, mode=mode, align_corners=align_corners, antialias=True)[0]

def get_polynomials_coefs_sqrtinv(max_degree):
    list_b = [1]
    temp = 1
    for n in range(1, max_degree+1):
        temp *= (2 * n - 1) / (2 * n)
        list_b.append(temp)

    return np.array(list_b, np.float32)


def load_geometry_from_sparse_voxels_model(sourse_path, load_attribute_field=True):
    voxel_model_frame = SparseVoxelModel(
        n_samp_per_vox=1,
        sh_degree=0,
        ss=1.0,
        white_background=False,
        black_background=True,
    )
    voxel_model_frame.load(sourse_path)

    geometry_info = {}
    geometry_info["morton_code_tensor"] = voxel_model_frame.vox_morton_code.clone().detach()
    geometry_info["vox_roots_tensor"] = voxel_model_frame.vox_root.clone().detach()
    geometry_info["vox_length_tensor"] = voxel_model_frame.vox_size.clone().detach()
    geometry_info["vox2corners_tensor"] = voxel_model_frame.vox_key.clone().detach()
    geometry_info["vox_params_geo_tensor"] = voxel_model_frame.vox_conners_geo_params.clone().detach()

    if load_attribute_field:
        geometry_info["vox_params_color_tensor"] = voxel_model_frame.vox_static_rgbs.clone().detach()

    del voxel_model_frame
    torch.cuda.empty_cache()

    return geometry_info


class DensityField:
    def __init__(
        self,
        source_path,
        num_sample_per_voxel=1,
        load_attribute_field=True
    ):
        # CONFIG INFO
        self.num_sample_per_voxel = num_sample_per_voxel

        geometry_info = load_geometry_from_sparse_voxels_model(source_path, load_attribute_field)
        #### Define the Density Field
        self.voxel_morton_code = geometry_info["morton_code_tensor"]
        self.voxel_position    = geometry_info["vox_roots_tensor"]
        self.voxel_size        = geometry_info["vox_length_tensor"]
        self.voxel2corner      = geometry_info["vox2corners_tensor"]
        self.corner_geo_params = geometry_info["vox_params_geo_tensor"]
        self.num_voxels        = self.voxel_morton_code.shape[0]

        #### Define current Attribute Field
        if load_attribute_field:
            self.voxel_attribute_field = geometry_info["vox_params_color_tensor"]
        else:
            self.voxel_attribute_field = torch.full(
                [self.num_voxels, 3], 0.0, 
                dtype=torch.float32
            )
    
    def apply_phi(self, view_info):

        ss = 1.5
        w_src, h_src = view_info.render_width, view_info.render_height
        w, h = round(w_src * ss), round(h_src * ss)
        w_ss, h_ss = w / w_src, h / h_src

        camera_settings = decompress_render.renderer.CameraSettings(
            image_width=w,#view_info.render_width, 
            image_height=h,#view_info.render_height, 
            tanfovx=view_info.tanfovx, 
            tanfovy=view_info.tanfovy, 
            cx=view_info.cx * w_ss,
            cy=view_info.cy * h_ss,
            w2c_matrix=view_info.world2view.cuda(),
            c2w_matrix=view_info.view2world.cuda()
        )

        render_settings = decompress_render.renderer.RenderSettings(
            num_sample_per_vox=self.num_sample_per_voxel,
            bg_color=0.0,
            near=view_info.near,
            need_depth=False,
            track_max_w=False,
            debug=False,
        )

        out_color, out_depth, out_T, max_w = decompress_render.renderer.rasterize_voxels_main(
            camera_settings=camera_settings,
            render_settings=render_settings,
            morton_code=self.voxel_morton_code,
            vox_roots=self.voxel_position,
            vox_length=self.voxel_size,
            vox2corners=self.voxel2corner,
            vox_params_geo=self.corner_geo_params,
            vox_params_color=self.voxel_attribute_field,
        )
        color = color + out_T * color.mean((1,2), keepdim=True)
        print(f"color.shape={color.shape}")
        color = resize_rendering(color, size=(h_src, w_src))

        rendered_image = torch.permute(out_color.clamp(0, 1), dims=(1, 2, 0)).detach()

        return rendered_image
    
    
    def apply_phi_with_F(self, voxel_attribute, view_info):

        if voxel_attribute.shape != (self.num_voxels, 3):
            raise Exception(f"Expect voxel_attribute in is static voxels color ({self.num_voxels}, 3) but got", voxel_attribute.shape)

        w, h = view_info.render_width, view_info.render_height
        camera_settings = decompress_render.renderer.CameraSettings(
            image_width=view_info.render_width, 
            image_height=view_info.render_height, 
            tanfovx=view_info.tanfovx, 
            tanfovy=view_info.tanfovy, 
            cx=view_info.cx,
            cy=view_info.cy,
            w2c_matrix=view_info.world2view.cuda(),
            c2w_matrix=view_info.view2world.cuda()
        )

        render_settings = decompress_render.renderer.RenderSettings(
            num_sample_per_vox=self.num_sample_per_voxel,
            bg_color=0.0,
            near=view_info.near,
            need_depth=False,
            track_max_w=False,
            debug=False,
        )

        out_color, _, _, _ = decompress_render.renderer.rasterize_voxels_main(
            camera_settings=camera_settings,
            render_settings=render_settings,
            morton_code=self.voxel_morton_code,
            vox_roots=self.voxel_position,
            vox_length=self.voxel_size,
            vox2corners=self.voxel2corner,
            vox_params_geo=self.corner_geo_params,
            vox_params_color=voxel_attribute, # use as input voxel_attribute 
        )

        rendered_image = torch.permute(out_color.clamp(0, 1), dims=(1, 2, 0))

        return rendered_image

    def apply_A_transpose(
            self,
            lowres_denfield,
            decoded_attribute_field,
            scene_dataset,
            optim_opt={}
        ):

        MSE_calculator = torch.nn.MSELoss()

        approx_attribute_field = torch.nn.Parameter(torch.full(
            [self.num_voxels, 3], 0.5, 
            dtype=torch.float32, device="cuda"
        ))
        optimizer = torch.optim.Adam([approx_attribute_field], lr=optim_opt.get("lr", 0.001))
        train_views = scene_dataset.get_train_views()
        test_views = scene_dataset.get_encoder_views()

        i=0
        for loop in range(optim_opt.get("num_scene_loop", 30)):
            k=0
            for view in train_views:
                ##
                optimizer.zero_grad()
                true_low_res_img   = lowres_denfield.apply_phi_with_F(decoded_attribute_field, view)
                approx_low_res_img = self.apply_phi_with_F(approx_attribute_field, view)
                ###
                mse_val = MSE_calculator(approx_low_res_img, true_low_res_img)
                mse_val.backward()
                ###

                if (i%500==0):
                    list_mse = []
                    for test_v in test_views:
                        true_low_res_img   = lowres_denfield.apply_phi_with_F(decoded_attribute_field, test_v).detach()
                        approx_low_res_img = self.apply_phi_with_F(approx_attribute_field, test_v).detach()
                        mse_test = MSE_calculator(approx_low_res_img, true_low_res_img).detach()
                        list_mse.append(mse_test.cpu())
                    print(f"Apply_A_transpose: loop = {loop, k} iter = {i} psnr={-10 * np.log10(np.mean(list_mse))}")
                    
                optimizer.step()
                i+=1
                k+=1

        output = approx_attribute_field.detach()
        list_mse = []
        for test_v in test_views:
            true_low_res_img   = lowres_denfield.apply_phi_with_F(decoded_attribute_field, test_v).detach()
            approx_low_res_img = self.apply_phi_with_F(output, test_v).detach()
            mse_test = MSE_calculator(approx_low_res_img, true_low_res_img).detach()
            list_mse.append(mse_test.cpu())
        print(f"Apply_A_transpose output: psnr={-10 * np.log10(np.mean(list_mse))}")
        
        # Cleaning
        optimizer.zero_grad()
        del optimizer
        torch.cuda.empty_cache()

        return output


    @torch.compiler.disable
    def apply_subview_phiTphi_with_F(self, voxel_attribute, view_info):

        w, h = view_info.render_width, view_info.render_height
        camera_settings = decompress_render.renderer.CameraSettings(
            image_width=view_info.render_width, 
            image_height=view_info.render_height, 
            tanfovx=view_info.tanfovx, 
            tanfovy=view_info.tanfovy, 
            cx=view_info.cx,
            cy=view_info.cy,
            w2c_matrix=view_info.world2view.cuda(),
            c2w_matrix=view_info.view2world.cuda()
        )

        render_settings = decompress_render.renderer.RenderSettings(
            num_sample_per_vox=self.num_sample_per_voxel,
            bg_color=0.0,
            near=view_info.near,
            need_depth=False,
            track_max_w=False,
            debug=False,
        )

        out_transformed_attribute = decompress_render.renderer.geometry_attention_transform_main(
            camera_settings=camera_settings,
            render_settings=render_settings,
            morton_code=self.voxel_morton_code,
            vox_roots=self.voxel_position,
            vox_length=self.voxel_size,
            vox2corners=self.voxel2corner,
            vox_params_geo=self.corner_geo_params,
            vox_params_attribute=voxel_attribute, # use as input voxel_attribute 
        )

        return out_transformed_attribute
    
    
    # @torch.compiler.disable
    def apply_full_phiTphi_with_F(self, voxel_attribute, scene_dataset):

        if voxel_attribute.shape != (self.num_voxels, 3):
            raise Exception(f"Expect voxel_attribute in is static ({self.num_voxels}, 3) but got", voxel_attribute.shape)

        list_views = scene_dataset.get_encoder_views()
        transformed_attribute = torch.full(
            [self.num_voxels, 3], 0.0, 
            dtype=torch.float32, device="cuda"
        )
        # vi = 0
        for view in list_views:
            # s = time.time()
            transformed_attribute = transformed_attribute + self.apply_subview_phiTphi_with_F(voxel_attribute, view)
            # print(f"++++ Apply_full_phiTphi_with_F: finished view {vi} in {time.time()-s}")
            # vi += 1

        return transformed_attribute


    # @torch.compiler.disable
    def apply_scaled_rotation_phiTphi(self, voxel_attribute, scene_dataset, rotation_opt={}, poly_coefs_sqrtinv=None):
        output = torch.full(
            [self.num_voxels, 3], 0.0, 
            dtype=torch.float32, device="cuda"
        )

        diagonal = self.apply_full_phiTphi_with_F(
            torch.ones_like(voxel_attribute), 
            scene_dataset
        ).detach()

        sqrtinv_diagonal = torch.where(
            diagonal>0.0,
            1.0 / torch.sqrt(diagonal),
            1.0
        )
        # sqrtinv_diagonal = torch.ones_like(diagonal)

        max_degree = rotation_opt.get("max_degree", 10)
        max_degree_grad = rotation_opt.get("max_degree_grad", 2)
        if poly_coefs_sqrtinv is None:
            poly_coefs_sqrtinv = get_polynomials_coefs_sqrtinv(max_degree)

        signal = voxel_attribute

        muy_phi = rotation_opt.get("muy_phi", 1.0)
        output = signal * poly_coefs_sqrtinv[0] * torch.tensor(np.sqrt(2*muy_phi))
        temp_val = signal
        
        for p in range(1, len(poly_coefs_sqrtinv)):
            s = time.time()

            if p <= max_degree_grad:
                Xb = self.apply_full_phiTphi_with_F(
                    temp_val*sqrtinv_diagonal,
                    scene_dataset
                ) * sqrtinv_diagonal
            else:
                with torch.no_grad():
                    Xb = self.apply_full_phiTphi_with_F(
                        temp_val*sqrtinv_diagonal,
                        scene_dataset
                    ) * sqrtinv_diagonal

            temp_val = temp_val - torch.tensor(2 * muy_phi) * Xb
            output = output + temp_val * poly_coefs_sqrtinv[p] * torch.tensor(np.sqrt(2*muy_phi))
            # print(f"+++ Apply_scaled_rotation_phiTphi: finished degree {p} in {time.time() - s}")

        output = output * sqrtinv_diagonal
        return output

    # @torch.compiler.disable
    # def apply_scaled_rotation_phiTphi_transpose(self, voxel_attribute, scene_dataset, rotation_opt={}):
    #     output = torch.full(
    #         [self.num_voxels, 3], 0.0, 
    #         dtype=torch.float32, device="cuda"
    #     )

    #     diagonal = self.apply_full_phiTphi_with_F(
    #         torch.ones_like(voxel_attribute), 
    #         scene_dataset
    #     ).detach()

    #     sqrtinv_diagonal = torch.where(
    #         diagonal>0.0,
    #         1.0 / torch.sqrt(diagonal),
    #         1.0
    #     )
    #     # sqrtinv_diagonal = torch.ones_like(diagonal)

    #     max_degree = rotation_opt.get("max_degree", 10)
    #     poly_coefs_sqrtinv = get_polynomials_coefs_sqrtinv(max_degree)
    #     signal = voxel_attribute * sqrtinv_diagonal

    #     muy_phi = rotation_opt.get("muy_phi", 1.0)
    #     output = signal * torch.tensor(poly_coefs_sqrtinv[0] * np.sqrt(2*muy_phi))
    #     temp_val = signal
        
    #     for p in range(1, max_degree):
    #         s = time.time()
    #         Xb = self.apply_full_phiTphi_with_F(
    #             temp_val*sqrtinv_diagonal,
    #             scene_dataset
    #         ) * sqrtinv_diagonal

    #         temp_val = temp_val - torch.tensor(2 * muy_phi) * Xb
    #         output = output + temp_val * torch.tensor(poly_coefs_sqrtinv[p] * np.sqrt(2*muy_phi))
    #         # print(f"+++ Apply_scaled_rotation_phiTphi: finished degree {p} in {time.time() - s}")

    #     output = output
    #     return output

    def decoding_attribute_field(self, encoded_coefs, scene_dataset, rotation_opt={}, poly_coefs_sqrtinv=None):
        with torch.no_grad():
            decoded_attribute_field = self.apply_scaled_rotation_phiTphi(
                encoded_coefs,
                scene_dataset,
                rotation_opt,
                poly_coefs_sqrtinv
            )
        return decoded_attribute_field

    def simulate_quantization(self, coefs, delta):

        with torch.no_grad():
            index_bins = (coefs / delta)
            ind_noise = torch.round(index_bins) - index_bins

        x_hat = coefs + ind_noise * delta
        return x_hat

    def encoding_attribute_field(self, enc_params, voxel_attribute, scene_dataset, rotation_opt={}, encoding_opts={}):

        print(f"Start Encoding Process custom_encoding_opts={encoding_opts}")
        MSE_calculator = torch.nn.MSELoss()
        dmode=encoding_opts.get("dmode", "coef_space")

        lambda_rate, log2_deltaQ, encoded_coefs, log2_spread_b, poly_coefs = enc_params
        lr = encoding_opts.get("lr", 0.001)
        if poly_coefs is None:
            optimizer = torch.optim.Adam([encoded_coefs, log2_spread_b, log2_deltaQ], lr=lr, eps=1e-20)
        else:
            optimizer = torch.optim.Adam([encoded_coefs, poly_coefs, log2_spread_b, log2_deltaQ], lr=lr, eps=1e-20)

        train_views = scene_dataset.get_train_views()
        test_views = scene_dataset.get_encoder_views()

        i=0
        num_loop = encoding_opts.get("num_loop", 30)
        for loop in range(num_loop):
            k=0
            for view in train_views:
                optimizer.zero_grad()

                decoded_attribute_field = self.apply_scaled_rotation_phiTphi(
                    self.simulate_quantization(encoded_coefs, torch.exp2(log2_deltaQ)),
                    scene_dataset,
                    rotation_opt,
                    poly_coefs_sqrtinv=poly_coefs
                )

                ###
                num_pixels = view.render_width * view.render_height
                if dmode=="render_space":
                    true_rendered_img_field   = self.apply_phi_with_F(voxel_attribute, view)
                    approx_rendered_img_field = self.apply_phi_with_F(decoded_attribute_field, view)
                    mse = MSE_calculator(true_rendered_img_field, approx_rendered_img_field)
                elif dmode=="coef_space":
                    mse = MSE_calculator(voxel_attribute, decoded_attribute_field)
                
                if lambda_rate > 0:
                    rate = (torch.abs(encoded_coefs) / (torch.exp2(log2_spread_b) * np.log(2)) + log2_spread_b - log2_deltaQ + 1)
                    loss = mse + lambda_rate * (rate.sum() / num_pixels)
                else:
                    loss = mse

                ###
                loss.backward()
                optimizer.step()
                if (i%100==0):
                    with torch.no_grad():
                        list_mse = []
                        
                        num_pixels = test_views[0].render_width * test_views[0].render_height
                        rate_test = (torch.abs(encoded_coefs) / (torch.exp2(log2_spread_b) * np.log(2)) + log2_spread_b - log2_deltaQ + 1)
                        mse_test_02 = MSE_calculator(voxel_attribute, decoded_attribute_field).detach()
                        for test_v in test_views:
                            true_rendered_img_field   = self.apply_phi_with_F(voxel_attribute, test_v).detach()
                            approx_rendered_img_field = self.apply_phi_with_F(decoded_attribute_field, test_v).detach()
                            mse_test_01 = MSE_calculator(true_rendered_img_field, approx_rendered_img_field).detach()
                            list_mse.append(mse_test_01.cpu())
                    print((
                        f"Encoding_attribute_field: TESTING loop = {loop, k} iter = {i} "
                        f"psnr={-10 * np.log10(np.mean(list_mse))} "
                        f"psnr_delta_field={-10 * np.log10(mse_test_02.cpu())} "
                        f"rate={rate_test.sum().detach() / num_pixels} "
                        f"spread={torch.exp2(log2_spread_b).detach().cpu().numpy()} "
                        f"deltaQ={torch.exp2(log2_deltaQ).detach().cpu().numpy()} "
                    ))

                ###
                i+=1
                k+=1

        output = encoded_coefs.detach()
        
        # Cleaning
        optimizer.zero_grad()
        del optimizer
        torch.cuda.empty_cache()

        return output

class MultiScaleSplatsEncoder:
    def __init__(
        self,
        scene_dataset_source_path,
        list_density_field_paths,
        deltaQ=0.005,
        lambda_rate=0.00001,
        list_spread_b_init=[1.0, 0.2, 0.085, 0.05, 0.035, 0.025, 0.02, 0.015], 
        num_sample_per_voxel=1,
        scene_loader_opt={
            "res_downscale": 8,
            "res_width": 0,
            "encoder_view_every": 20
        },
        A_transpose_opt={
            "lr": 0.001,
            "num_scene_loop": 30,
        },
        rotation_opt={
            "max_degree": 5,
            "muy_phi": 1.0,
        },
        encoding_opts={
            "lr": 0.005,
            "num_loop": 20,
        },
        lowpass_rotation_opt = {
            "max_degree": 2,
            "muy_phi": 0.9,
        },
        lowpass_encoding_opts={
            "lr": 0.005,
            "num_loop": 45,
        }
    ):

        self.list_density_field = []
        for path in list_density_field_paths:
            self.list_density_field.append(
                DensityField(path, num_sample_per_voxel, load_attribute_field=True)
            )

        self.scene_dataset = Scene(
            scene_dataset_source_path, image_dir_name="images",
            res_downscale=scene_loader_opt["res_downscale"],
            res_width=scene_loader_opt["res_width"],
            encoder_view_every=scene_loader_opt["encoder_view_every"],
        )
        self.lambda_rate = lambda_rate

        self.lowpass_rotation_opt  = lowpass_rotation_opt
        self.lowpass_encoding_opts = lowpass_encoding_opts
        self.A_transpose_opt       = A_transpose_opt
        self.rotation_opt          = rotation_opt
        self.encoding_opts         = encoding_opts

        ###############################################################
        ###############################################################
        ###############################################################
        
        list_learned_coefs = []
        list_learned_logspread = []
        list_poly_coefs = []
        L = len(self.list_density_field)
        for i in range(L):
            if i == 0:
                list_poly_coefs.append(torch.nn.Parameter(torch.tensor(
                    get_polynomials_coefs_sqrtinv(self.lowpass_rotation_opt["max_degree"]),
                    dtype=torch.float32, device="cuda"
                )))
            else:
                list_poly_coefs.append(torch.nn.Parameter(torch.tensor(
                    get_polynomials_coefs_sqrtinv(self.rotation_opt["max_degree"]),
                    dtype=torch.float32, device="cuda"
                )))

            list_learned_coefs.append(
                torch.nn.Parameter(torch.full(
                    [self.list_density_field[i].num_voxels, 3], 0.0, 
                    dtype=torch.float32, device="cuda"
                ))
            )
            list_learned_logspread.append(
                torch.nn.Parameter(torch.full(
                    [1, 3], np.log2(list_spread_b_init[i]), 
                    dtype=torch.float32, device="cuda"
                ))
            )

        self.trainable_params = torch.nn.ParameterDict({
            "list_poly_coefs"        : torch.nn.ParameterList(list_poly_coefs),
            "list_learned_coefs"     : torch.nn.ParameterList(list_learned_coefs),
            "list_learned_logspread" : torch.nn.ParameterList(list_learned_logspread),
            "log2_deltaQ" : torch.nn.Parameter(
                torch.full(
                [1], np.log2(deltaQ), 
                dtype=torch.float32, device="cuda"
            ))       
        })


    def quantize(self, coef, deltaQ):
        with torch.no_grad():
            index = torch.round(coef / deltaQ)
            quantized_coefs = index * deltaQ
            
        return quantized_coefs

    def learn_encoding(self):
        list_quantized_coefs = []
        L = len(self.list_density_field)

        #### Lowpass
        lowest_field = self.list_density_field[0]
        tobe_enc_param = (
            self.lambda_rate, 
            self.trainable_params["log2_deltaQ"], 
            self.trainable_params["list_learned_coefs"][0], 
            self.trainable_params["list_learned_logspread"][0],
            self.trainable_params["list_poly_coefs"][0]
        )
        encoded_coefs = lowest_field.encoding_attribute_field(
            tobe_enc_param,
            lowest_field.voxel_attribute_field,
            self.scene_dataset,
            rotation_opt=self.lowpass_rotation_opt,
            encoding_opts=self.lowpass_encoding_opts,
        )
        # Quantizer
        quantized_coefs = self.quantize(encoded_coefs, torch.exp2(self.trainable_params["log2_deltaQ"]))
        list_quantized_coefs.append(
            quantized_coefs
        )
        decoded_attribute_field = lowest_field.decoding_attribute_field(
            quantized_coefs,
            self.scene_dataset,
            rotation_opt=self.rotation_opt,
            poly_coefs_sqrtinv=self.trainable_params["list_poly_coefs"][0]
        )
        for i in range(1, L, 1):
            print(f"Encoding level={i}")
            lowres_denfield = self.list_density_field[i-1]
            highres_denfield = self.list_density_field[i]

            # A^T
            approx_attribute_field = highres_denfield.apply_A_transpose(
                lowres_denfield,
                decoded_attribute_field,
                self.scene_dataset,
                optim_opt=self.A_transpose_opt
            )

            # Encoding residual gap 
            residual_gap_true = highres_denfield.voxel_attribute_field - approx_attribute_field
            tobe_enc_param = (
                self.lambda_rate, 
                self.trainable_params["log2_deltaQ"], 
                self.trainable_params["list_learned_coefs"][i], 
                self.trainable_params["list_learned_logspread"][i],
                self.trainable_params["list_poly_coefs"][i]
            )
            encoded_coefs = highres_denfield.encoding_attribute_field(
                tobe_enc_param,
                residual_gap_true,
                self.scene_dataset,
                rotation_opt=self.rotation_opt,
                encoding_opts=self.encoding_opts
            )

            # Quantizer
            quantized_coefs = self.quantize(encoded_coefs, torch.exp2(self.trainable_params["log2_deltaQ"]))
            list_quantized_coefs.append(
                quantized_coefs
            )

            #Decoding Residual Gap
            residual_gap_decoded = highres_denfield.decoding_attribute_field(
                quantized_coefs,
                self.scene_dataset,
                rotation_opt=self.rotation_opt,
                poly_coefs_sqrtinv=self.trainable_params["list_poly_coefs"][i]
            )
            decoded_attribute_field = approx_attribute_field + residual_gap_decoded

        return list_quantized_coefs, decoded_attribute_field
    

    def extract_learned_encoding(self):

        list_quantized_coefs = []
        L = len(self.list_density_field)


        lowest_field = self.list_density_field[0]
        encoded_coefs = self.trainable_params["list_learned_coefs"][0].detach()
        # Quantizer
        quantized_coefs = self.quantize(encoded_coefs, torch.exp2(self.trainable_params["log2_deltaQ"]))
        list_quantized_coefs.append(
            quantized_coefs
        )
        decoded_attribute_field = lowest_field.decoding_attribute_field(
            quantized_coefs,
            self.scene_dataset,
            rotation_opt=self.rotation_opt,
            poly_coefs_sqrtinv=self.trainable_params["list_poly_coefs"][0]
        )

        for i in range(1, L, 1):
            print(f"Extract encoding level={i}")
            lowres_denfield = self.list_density_field[i-1]
            highres_denfield = self.list_density_field[i]

            # A^T
            approx_attribute_field = highres_denfield.apply_A_transpose(
                lowres_denfield,
                decoded_attribute_field,
                self.scene_dataset,
                optim_opt=self.A_transpose_opt
            )

            # Encoding residual gap 
            encoded_coefs = self.trainable_params["list_learned_coefs"][i].detach()

            # Quantizer
            quantized_coefs = self.quantize(encoded_coefs, torch.exp2(self.trainable_params["log2_deltaQ"]))
            list_quantized_coefs.append(
                quantized_coefs
            )

            #Decoding Residual Gap
            residual_gap_decoded = highres_denfield.decoding_attribute_field(
                quantized_coefs,
                self.scene_dataset,
                rotation_opt=self.rotation_opt,
                poly_coefs_sqrtinv=self.trainable_params["list_poly_coefs"][i]
            )
            decoded_attribute_field = approx_attribute_field + residual_gap_decoded

        return list_quantized_coefs, decoded_attribute_field

    

# class MultiScaleSplatsDecoder:
#     def __init__(
#         self,
#         scene_dataset_source_path,
#         list_density_field_paths,
#         num_sample_per_voxel=1,
#         scene_loader_opt={
#             "res_downscale": 8.0,
#             "res_width": 0,
#             "encoder_view_every": 20,
#         },
#         A_transpose_opt={
#             "lr": 0.001,
#             "num_scene_loop": 20,
#         },
#         rotation_opt={
#             "max_degree":10,
#             "muy_phi": 1.0,
#         }
#     ):
        
#         self.list_density_field = []
#         for path in list_density_field_paths:
#             self.list_density_field.append(
#                 DensityField(path, num_sample_per_voxel, load_attribute_field=False)
#             )

#         self.scene_dataset = Scene(
#             scene_dataset_source_path, image_dir_name="images",
#             res_downscale=scene_loader_opt["res_downscale"],
#             res_width=scene_loader_opt["res_width"],
#             encoder_view_every=scene_loader_opt["encoder_view_every"],
#         )

#         self.A_transpose_opt=A_transpose_opt
#         self.rotation_opt=rotation_opt
#         self.lowpass_rotation_opt = {
#                 "max_degree":2,
#                 "muy_phi": 0.9,
#         }
    
#     def decoding(self, list_quantized_coefs):
        
#         L = len(self.list_density_field)
#         lowest_field = self.list_density_field[0]
#         decoded_attribute_field = lowest_field.decoding_attribute_field(
#             list_quantized_coefs[0],
#             self.scene_dataset,
#             rotation_opt=self.lowpass_rotation_opt
#         )

#         for i in range(1, L, 1):
#             print(f"Decoding level={i}")
#             lowres_denfield = self.list_density_field[i-1]
#             highres_denfield = self.list_density_field[i]

#             # A^T
#             approx_attribute_field = highres_denfield.apply_A_transpose(
#                 lowres_denfield,
#                 decoded_attribute_field,
#                 self.scene_dataset,
#                 optim_opt=self.A_transpose_opt
#             )

#             #Decoding Residual Gap
#             residual_gap_decoded = highres_denfield.decoding_attribute_field(
#                 list_quantized_coefs[i],
#                 self.scene_dataset,
#                 rotation_opt=self.rotation_opt
#             )
#             decoded_attribute_field = approx_attribute_field + residual_gap_decoded
        
#         return decoded_attribute_field
