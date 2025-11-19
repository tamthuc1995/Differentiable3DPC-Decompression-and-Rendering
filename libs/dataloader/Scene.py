import os
import json
import time 
import natsort
import pycolmap
import numpy as np
import torch

from PIL import Image
from pathlib import Path
from typing import NamedTuple

from libs.dataloader.View import View, ViewCreator


class PointCloud(NamedTuple):
    positions: np.array
    rgb: np.array
    err: np.array


class Scene:
    def __init__(
        self,
        source_path, image_dir_name="images",
        res_downscale=0.0, res_width=0,
        test_every=8
    ):

        sparse_path = os.path.join(source_path, "sparse")
        colmap_path = os.path.join(source_path, "colmap", "sparse")

        if os.path.exists(sparse_path) or os.path.exists(colmap_path):
            print("Read dataset in COLMAP format.")
            point_cloud, views2points, views_construct_list = read_from_colmap_dataset(
                source_path=source_path,
                image_dir_name=image_dir_name
            )
        else:
            raise Exception("Unknown scene type!")
        
        self.point_cloud = point_cloud
        self.views2points = views2points # dict: (view_name, points_ind)

        self.viewDataTrain = []
        self.viewDataTest  = []
        view_creator = ViewCreator(
            res_downscale=res_downscale,
            res_width=res_width,
        )
        for view_i, viewInfo in enumerate(views_construct_list):
            # s = time.time()
            if (test_every>0) and (view_i % test_every ==0):
                self.viewDataTest.append(view_creator(**viewInfo))
                # print(f"Done view {viewInfo["view_name"]} for testing in {time.time()-s}s")
            else:
                self.viewDataTrain.append(view_creator(**viewInfo))
                # print(f"Done view {viewInfo["view_name"]} for training in {time.time()-s}s")
                
    def get_train_views(self):
        return self.viewDataTrain
    
    def get_test_views(self):
        return self.viewDataTest



########################################################
######## Extract views data from COLMAP         ########
########################################################
def read_from_colmap_dataset(source_path, image_dir_name):
    source_path = Path(source_path)

    # Parse colmap meta data
    sparse_path = source_path / "sparse" / "0"
    if not sparse_path.exists():
        sparse_path = source_path / "colmap" / "sparse" / "0"
    if not sparse_path.exists():
        raise Exception("Can not find COLMAP reconstruction.")
    

    # load COLMAP data
    SfM = pycolmap.Reconstruction(sparse_path)

    ########################################################
    ########          Extract point cloud           ########
    ########################################################
    pc_positions = []
    pc_rgb = []
    pc_err = []
    p_id = []
    num_p = 0
    for k, v in SfM.points3D.items():
        p_id.append(k)
        pc_positions.append(v.xyz)
        pc_rgb.append(v.color)
        pc_err.append(v.error)
        num_p += 1

    pc_positions = np.array(pc_positions)
    pc_rgb = np.array(pc_rgb)
    pc_err = np.array(pc_err)
    p_id = np.array(p_id)

    points_idmap = np.full([p_id.max()+2], -1, dtype=np.int64)
    points_idmap[p_id] = np.arange(num_p)

    views2points = {}
    for image in SfM.images.values():
        matchs = np.array([p.point3D_id for p in image.points2D if p.has_point3D()])
        matched_p_id = points_idmap[matchs]
        assert matched_p_id.min() >= 0 and matched_p_id.max() < num_p
        views2points[image.name] = matched_p_id

    point_cloud = PointCloud(
        positions=pc_positions,
        rgb=pc_rgb,
        err=pc_err,
    )

    ########################################################
    ######## Extract views data from camera/images  ########
    ########################################################
    # Sort key by filename
    keys = sorted(SfM.images.keys(), key = lambda k : SfM.images[k].name)


    # Load all images and cameras
    views_construct_list = []
    for key in keys:

        view = SfM.images[key]
        # Load image
        image_path = source_path / image_dir_name / view.name
        if not image_path.exists():
            raise Exception(f"File not found: {str(image_path)}")
        image = Image.open(image_path)

        # Load camera intrinsic
        if view.camera.model.name == "SIMPLE_PINHOLE":
            focal_x, cx, cy = view.camera.params
            fovx = 2 * np.arctan(view.camera.width / (2 * focal_x))
            fovy = 2 * np.arctan(view.camera.width / (2 * focal_x))
            cx_p = cx / view.camera.width
            cy_p = cy / view.camera.height
        elif view.camera.model.name == "PINHOLE":
            focal_x, focal_y, cx, cy = view.camera.params
            fovx = 2 * np.arctan(view.camera.width / (2 * focal_x))
            fovy = 2 * np.arctan(view.camera.width / (2 * focal_y))
            cx_p = cx / view.camera.width
            cy_p = cy / view.camera.height
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # Load camera extrinsic
        world2view = np.eye(4, dtype=np.float32)
        try:
            world2view[:3] = view.cam_from_world().matrix()
        except:
            # Older version of pycolmap
            world2view[:3] = view.cam_from_world.matrix()

        views_construct_list.append(dict(
            image=image,
            world2view=world2view,
            fovx=fovx,
            fovy=fovy,
            cx_p=cx_p,
            cy_p=cy_p,
            view_name=view.name,
        ))
    
    return point_cloud, views2points, views_construct_list

