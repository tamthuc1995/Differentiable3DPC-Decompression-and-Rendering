import torch
import numpy as np



class View:
    '''
    Base class of perspective cameras - View.
    '''
    def __init__(
        self, world2view, fovx, fovy, cx_p, cy_p,
        rendered_view, view_name,
        near=0.02,
    ):

        self.view_name = view_name

        # Camera/View parameters
        self.world2view = torch.tensor(world2view, dtype=torch.float32)
        self.view2world = self.world2view.inverse().contiguous()

        self.fovx = fovx
        self.fovy = fovy

        # Convert image to rendered view tensor
        self.rendered_true = torch.tensor(rendered_view, dtype=torch.float32)

        # Other camera parameters
        self.render_width = self.rendered_true.shape[2]
        self.render_height = self.rendered_true.shape[1]
        self.cx_p = cx_p
        self.cy_p = cy_p
        self.near = near


    def __repr__(self):
        clsname = self.__class__.__name__
        fname = f"from_image_name='{self.view_name}'"
        res = f"HW=({self.render_height}x{self.render_width})"
        fov = f"fovx={np.rad2deg(self.fovx):.1f}deg"
        return f"{clsname}({fname}, {res}, {fov})"
    
    def to(self, device):
        self.rendered_true = self.rendered_true.to(device)
        self.world2view = self.world2view.to(device)
        self.view2world = self.view2world.to(device)
        self.sparse_pt = self.sparse_pt.to(device)
        return self

    @property
    def lookat(self):
        return self.view2world[:3, 2]

    @property
    def position(self):
        return self.view2world[:3, 3]

    @property
    def down(self):
        return self.view2world[:3, 1]

    @property
    def right(self):
        return self.view2world[:3, 0]

    @property
    def cx(self):
        return self.render_width * self.cx_p

    @property
    def cy(self):
        return self.render_height * self.cy_p

    @property
    def pix_size(self):
        return 2 * self.tanfovx / self.render_width

    @property
    def tanfovx(self):
        return np.tan(self.fovx * 0.5)

    @property
    def tanfovy(self):
        return np.tan(self.fovy * 0.5)
    

    def project2view(self, pc_positions, return_depth=False):
        # Return normalized image coordinate in [-1, 1]
        view_coor = pc_positions @ self.world2view[:3, :3].T + self.world2view[:3, 3]
        depth = view_coor[:, [2]]
        image_uv = view_coor[:, :2] / depth
        scale_x = 1 / self.tanfovx
        scale_y = 1 / self.tanfovy
        shift_x = 2 * self.cx_p - 1
        shift_y = 2 * self.cy_p - 1
        image_uv[:, 0] = image_uv[:, 0] * scale_x + shift_x
        image_uv[:, 1] = image_uv[:, 1] * scale_y + shift_y
        if return_depth:
            return image_uv, depth
        return image_uv

# Function that create Camera instances while parsing dataset
class ViewCreator:
    def __init__(self, res_downscale=0.0, res_width=0):
        self.res_downscale = res_downscale
        self.res_width = res_width

    def __call__(self, image, world2view, fovx, fovy, cx_p=0.5, cy_p=0.5, view_name=""):

        # Get target resolution
        if self.res_downscale > 0:
            downscale = self.res_downscale
        elif self.res_width > 0:
            downscale = image.size[0] / self.res_width
        else:
            downscale = 1

        # Resize image if needed
        if downscale != 1:
            size = (round(image.size[0] / downscale), round(image.size[1] / downscale))
            image = image.resize(size)

        # Convert image to tensor
        img_arr = np.moveaxis(np.array(image), -1, 0) / 255.0
        if img_arr.shape[0] == 4:
            img_arr, alpha_channel = img_arr[:3], img_arr[[3]]


        return View(
            world2view=world2view,
            fovx=fovx, fovy=fovy,
            cx_p=cx_p, cy_p=cy_p,
            rendered_view=img_arr,
            view_name=view_name
        )
