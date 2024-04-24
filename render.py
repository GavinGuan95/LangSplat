#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.camera_utils import ns_cameras_to_langsplat_cameras
from nerfstudio.cameras.camera_paths import get_path_from_json
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from autoencoder.model import Autoencoder
import json

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_pt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_pt")
    gts_pt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_pt")

    makedirs(render_pt_path, exist_ok=True)
    makedirs(gts_pt_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
        

        #autoencoder = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to("cuda:0")
        #autoencoder.load_state_dict(torch.load("ckpt/sofa/test/autoencoder_ckpt.pth"))

        #language_feature_image = output["language_feature_image"].permute(1, 2, 0)
        # Initialize to empty tensor
        #ori_language_feature_image = autoencoder.decode(language_feature_image).permute(2, 0, 1)
        
        if not args.include_feature and not args.ignore_gt:
            gt = view.original_image[0:3, :, :]
            
        elif not args.ignore_gt:
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)

        torch.save(rendering.permute(1,2,0), os.path.join(render_pt_path, '{0:05d}'.format(idx) + ".pt"))
        if not args.ignore_gt:
            torch.save(gt.permute(1,2,0), os.path.join(gts_pt_path, '{0:05d}'.format(idx) + ".pt"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if not args.ignore_gt:
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, render_num_images=-1):
    if not args.render_camera_path:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, shuffle=False)
            checkpoint = os.path.join(args.model_path, 'chkpnt' + str(args.checkpoint_number) + '.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if not skip_train:
                render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(render_num_images=render_num_images), gaussians, pipeline, background, args)

            if not skip_test:
                render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(render_num_images=render_num_images), gaussians, pipeline, background, args)
    else:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)
            checkpoint = os.path.join(args.model_path, 'chkpnt' + str(args.checkpoint_number) + '.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            camera_path_json = json.load(open(args.camera_path_file))
            ns_cameras = get_path_from_json(camera_path_json)
            views = ns_cameras_to_langsplat_cameras(ns_cameras)
            render_set(dataset.model_path, dataset.source_path, "camera_path", None, views, gaussians, pipeline, background, args)



if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--checkpoint_number", default=30000, type=int)
    parser.add_argument("--render_camera_path", type=bool, default=False)
    parser.add_argument("--camera_path_file", type=str, default=None)
    parser.add_argument("--ignore_gt", type=bool, default=False)
    parser.add_argument("--render_num_images", type=int, default=-1)


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args, render_num_images=args.render_num_images)