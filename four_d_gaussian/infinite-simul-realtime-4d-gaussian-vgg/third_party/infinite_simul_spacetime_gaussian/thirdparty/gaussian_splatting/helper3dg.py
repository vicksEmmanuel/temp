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
import os
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json

import numpy as np 
import cv2
from tqdm import tqdm
import shutil

sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def getparser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) #we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10000, 12000, 25_000, 30_000])
    parser.add_argument("--test_iterations", default=-1, type=int)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5, help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default = "gaussian")
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v2")
    parser.add_argument("--configpath", type=str, default = "None")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    

    return args, lp.extract(args), op.extract(args), pp.extract(args)

def getrenderparts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]




def gettestparse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v3")
    parser.add_argument("--valloader", type=str, default = "colmap")
    parser.add_argument("--configpath", type=str, default = "1")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # configpath
    safe_state(args.quiet)
    
    multiview = True if args.valloader.endswith("mv") else False

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
        print("args: " + str(args))
        
    return args, model.extract(args), pipeline.extract(args), multiview
    
def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --SiftExtraction.use_gpu 0 --database_path " + dbfile+ " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)


    featurematcher = "colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)





def getcolmapsingleimundistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --SiftExtraction.use_gpu 0 --SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


 

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    #Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
   



def getcolmapsingleimdistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --SiftExtraction.use_gpu 0 --SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 
    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
        

def getcolmapsingletechni(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --SiftExtraction.use_gpu 0 --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)


    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
    
    return 


# =============================================================================
# CUT3R-based triangulation (alternative to COLMAP)
# =============================================================================
import struct
import collections as _collections

# CUT3R paths
_CUT3R_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "..", "..", "..", "..", "..", "..", "CUT3R")
_CUT3R_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "..", "..", "..", 
                                    "checkpoints", "cut3r", "cut3r_512_dpt_4_64.pth")

# Module-level singleton for CUT3R model (avoids reloading per frame)
_cut3r_model_cache = {"model": None, "path": None}


def _get_cut3r_model(model_path=None, device="cuda"):
    """Load CUT3R model with caching to avoid reloading per-frame."""
    if model_path is None:
        model_path = _CUT3R_DEFAULT_CKPT
    model_path = os.path.abspath(model_path)
    
    if _cut3r_model_cache["model"] is not None and _cut3r_model_cache["path"] == model_path:
        return _cut3r_model_cache["model"]
    
    # Set up CUT3R imports
    cut3r_root = os.path.abspath(_CUT3R_ROOT)
    if cut3r_root not in sys.path:
        sys.path.insert(0, cut3r_root)
    
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(model_path)
    from src.dust3r.model import ARCroco3DStereo
    
    print(f"[CUT3R] Loading model from {model_path}...")
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    
    _cut3r_model_cache["model"] = model
    _cut3r_model_cache["path"] = model_path
    print("[CUT3R] Model loaded and cached")
    return model


def _write_cameras_bin(cameras, path):
    """Write COLMAP cameras.bin. cameras: list of {id, width, height, fx, fy, cx, cy}"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<I", cam["id"]))
            f.write(struct.pack("<i", 1))  # PINHOLE model
            f.write(struct.pack("<Q", int(cam["width"])))
            f.write(struct.pack("<Q", int(cam["height"])))
            for p in [cam["fx"], cam["fy"], cam["cx"], cam["cy"]]:
                f.write(struct.pack("<d", p))


def _write_images_bin(images, path):
    """Write COLMAP images.bin. images: list of {id, qvec(4), tvec(3), camera_id, name}"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I", img["id"]))
            for q in img["qvec"]:
                f.write(struct.pack("<d", q))
            for t in img["tvec"]:
                f.write(struct.pack("<d", t))
            f.write(struct.pack("<I", img["camera_id"]))
            for ch in img["name"]:
                f.write(struct.pack("<c", ch.encode("utf-8")))
            f.write(struct.pack("<c", b"\x00"))
            f.write(struct.pack("<Q", 0))  # 0 points2D


def _write_points3D_bin(points3D, path):
    """Write COLMAP points3D.bin. points3D: dict of {id: {xyz(3), rgb(3), error}}"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3D)))
        for pid, pt in points3D.items():
            f.write(struct.pack("<Q", pid))
            for v in pt["xyz"]:
                f.write(struct.pack("<d", v))
            for c in pt["rgb"]:
                f.write(struct.pack("<B", int(c)))
            f.write(struct.pack("<d", pt.get("error", 0.0)))
            f.write(struct.pack("<Q", 0))  # 0 track length


def _rotmat2qvec_helper(R):
    """Convert rotation matrix to COLMAP quaternion [w, x, y, z]."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def getcut3rsinglen3d(folder, offset, model_path=None, device="cuda", size=512,
                      max_points_per_view=50000):
    """CUT3R-based per-frame triangulation — drop-in alternative to getcolmapsinglen3d.
    
    Instead of running COLMAP feature extraction, matching, and point triangulation,
    this function uses CUT3R's dense regression to predict 3D points for each pixel.
    
    Args:
        folder: Base dataset folder (parent of colmap_N directories)
        offset: Frame offset (colmap_N)
        model_path: Path to CUT3R checkpoint (uses default if None)
        device: Torch device
        size: CUT3R input image size
        max_points_per_view: Maximum dense points per camera view
    """
    import torch as _torch
    
    colmap_dir = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(colmap_dir), f"Missing colmap directory: {colmap_dir}"
    
    inputimagefolder = os.path.join(colmap_dir, "input")
    assert os.path.exists(inputimagefolder), f"Missing input images: {inputimagefolder}"
    
    sparse_out = os.path.join(colmap_dir, "sparse", "0")
    os.makedirs(sparse_out, exist_ok=True)
    
    # Read poses from manual/ (written during preprocess.py's convert_dnerf_to_colmap_db)
    manual_dir = os.path.join(colmap_dir, "manual")
    
    # Collect input images
    img_files = sorted([f for f in os.listdir(inputimagefolder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not img_files:
        print(f"[CUT3R] No images found in {inputimagefolder}")
        return
    
    img_paths = [os.path.join(inputimagefolder, f) for f in img_files]
    print(f"[CUT3R] Processing {len(img_paths)} images for offset {offset}")
    
    # Load CUT3R model (cached)
    model = _get_cut3r_model(model_path, device)
    
    # Set up CUT3R imports (already on path from model loading)
    from src.dust3r.utils.image import load_images_512
    from src.dust3r.inference import inference
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf
    
    # Prepare CUT3R input views
    # Use load_images_512 to force all images to 512x384, avoiding tensor size
    # mismatches when input images have different aspect ratios.
    images = load_images_512(img_paths, size=size)
    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"],
            "ray_map": _torch.full(
                (images[i]["img"].shape[0], 6,
                 images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                _torch.nan,
            ),
            "true_shape": _torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": _torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": _torch.tensor(True).unsqueeze(0),
            "ray_mask": _torch.tensor(False).unsqueeze(0),
            "update": _torch.tensor(True).unsqueeze(0),
            "reset": _torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    
    # Run CUT3R inference
    with _torch.no_grad():
        outputs, _ = inference(views, model, device)
    
    # Extract results
    pts3d_self = _torch.cat([o["pts3d_in_self_view"].cpu() for o in outputs["pred"]], 0)
    conf_self = _torch.cat([o["conf_self"].cpu() for o in outputs["pred"]], 0)
    
    pr_poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
                for pred in outputs["pred"]]
    c2w_all = _torch.cat(pr_poses, 0)  # (B, 4, 4)
    
    # Transform to world coordinates
    pts3d_world_ls = []
    pts3d_self_ls = [o["pts3d_in_self_view"].cpu() for o in outputs["pred"]]
    for pose, pself in zip(pr_poses, pts3d_self_ls):
        pts3d_world_ls.append(geotrf(pose, pself.unsqueeze(0) if pself.dim() == 3 else pself))
    pts3d_world = _torch.cat(pts3d_world_ls, 0)
    
    # Estimate focal lengths
    B, H, W, _ = pts3d_self.shape
    pp = _torch.tensor([W // 2, H // 2]).float().repeat(B, 1)
    focals = estimate_focal_knowing_depth(pts3d_self, pp, focal_mode="weiszfeld")
    
    # Get original image dimensions
    first_img = cv2.imread(img_paths[0])
    orig_h, orig_w = first_img.shape[:2]
    
    # Extract colors
    colors_all = _torch.cat([
        0.5 * (o["img"].permute(0, 2, 3, 1).cpu() + 1.0)
        for o in outputs["views"]
    ], 0)  # (B, H, W, 3)
    
    # Build COLMAP binary output
    colmap_cameras = []
    colmap_images = []
    all_points = {}
    point_id = 1
    
    for i in range(B):
        cam_name = img_files[i]
        c2w = c2w_all[i].numpy()
        
        # Scale focal to original image dimensions  
        f_cut3r = focals[i].item()
        scale_w = orig_w / W
        scale_h = orig_h / H
        fx = f_cut3r * scale_w
        fy = f_cut3r * scale_h
        cx = orig_w / 2.0
        cy = orig_h / 2.0
        
        colmap_cameras.append({
            "id": i + 1,
            "width": orig_w,
            "height": orig_h,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        })
        
        # c2w → w2c for COLMAP
        w2c = np.linalg.inv(c2w)
        qvec = _rotmat2qvec_helper(w2c[:3, :3])
        
        colmap_images.append({
            "id": i + 1,
            "qvec": qvec,
            "tvec": w2c[:3, 3],
            "camera_id": i + 1,
            "name": cam_name,
        })
        
        # Subsample dense points
        pts_view = pts3d_world[i].reshape(-1, 3)
        cols_view = colors_all[i].reshape(-1, 3)
        conf_view = conf_self[i].reshape(-1)
        
        valid = _torch.isfinite(pts_view).all(dim=-1) & _torch.isfinite(conf_view) & (conf_view > 0)
        pts_v = pts_view[valid]
        cols_v = cols_view[valid]
        conf_v = conf_view[valid]
        
        if len(pts_v) > 0:
            # Confidence filter
            conf_thr = _torch.quantile(conf_v, 0.5)
            mask = conf_v >= conf_thr
            pts_v = pts_v[mask]
            cols_v = cols_v[mask]
            conf_v = conf_v[mask]
            
            # Random subsample
            if len(pts_v) > max_points_per_view:
                weights = conf_v / conf_v.sum()
                indices = _torch.multinomial(weights, max_points_per_view, replacement=False)
                pts_v = pts_v[indices]
                cols_v = cols_v[indices]
            
            xyz_np = pts_v.numpy()
            rgb_np = (cols_v.numpy() * 255).clip(0, 255).astype(np.uint8)
            
            for j in range(len(xyz_np)):
                all_points[point_id] = {
                    "xyz": xyz_np[j],
                    "rgb": rgb_np[j],
                    "error": 0.0,
                }
                point_id += 1
    
    print(f"[CUT3R] offset={offset}: {len(all_points)} points from {B} views")
    
    # Write COLMAP binary files
    _write_cameras_bin(colmap_cameras, os.path.join(sparse_out, "cameras.bin"))
    _write_images_bin(colmap_images, os.path.join(sparse_out, "images.bin"))
    _write_points3D_bin(all_points, os.path.join(sparse_out, "points3D.bin"))
    
    # Copy images to colmap_N/images/ (the undistorted images directory)
    images_out = os.path.join(colmap_dir, "images")
    os.makedirs(images_out, exist_ok=True)
    for f in img_files:
        src = os.path.join(inputimagefolder, f)
        dst = os.path.join(images_out, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Remove input directory (matches COLMAP behavior)
    if os.path.exists(inputimagefolder):
        shutil.rmtree(inputimagefolder)
    
    print(f"[CUT3R] ✅ Triangulation complete for offset {offset}")

