# Optimized Gaussian Splatting Run
```bash
nohup python3 -u orchestrator.py \
    --video_folder /workspace/sim-animate-environment/outputs/generated_scene_01 \
    --scene_name "generated_scene_01" \
    --num_videos 15 \
    --end_frame 50 \
    --v2v_model cosmos \
    --use_vggt \
    > orchestrator_run.log 2>&1 &
```



```bash
nohup python3 -u orchestrator.py \
    --image /workspace/sim-animate-environment/test/prototype_image_2.jpg \
    --prompt "Lightning strike in the scenery" \
    --scene_name "generated_scene_01" \
    --num_videos 20 \
    --end_frame 50 \
    --use_vggt \
    --v2v_model cosmos \
    > orchestrator_run.log 2>&1 &
```




Low grade push

```
<!-- @format -->

# limit threads to one (already done, but repeat just in case)

git config pack.threads 1

# cap the amount of heap Git will allocate while packing

# (10 MB is very low – bump it if you have a little more RAM)

git config pack.windowMemory 10m
git config pack.packSizeLimit 20m

# or, for a one‑off push, use the equivalent options on git-repack:

git repack -a -d --window-memory=10m --max-pack-size=20m

# then push as usual

git push -u origin main

```





### POSES Generator

```bash
# Processes all frames from each video automatically
python3 generate_poses_ttt3r.py --folder /workspace/sim-animate-environment/test/environment1
```


python3 generate_poses_ttt3r.py --folder /workspace/sim-animate-environment/test/environment1 --use_colmap_poses


```bash
# Processes all frames from each video automatically (best for dynamic 4D scenes)
python3 generate_poses_sharp.py --folder /workspace/sim-animate-environment/test/coffee_martini --output coffee_martini_sharp.npy

# NOTE: If your scene appears upside-down in the trainer, try adding --no_axis_flip
```

### 3D Model Generator (Point Cloud)

```bash
# Generate 3D model from a single video (temporal trajectory)
python3 generate_model_ttt3r.py --source /workspace/sim-animate-environment/test/environment1/cam_00.mp4 --output environment_model.ply

# Generate 3D model from a folder of multi-camera videos
python3 generate_model_ttt3r.py --source /workspace/sim-animate-environment/test/environment1 --output scene_model.ply
```





### Compare Poses

```bash
python3 compare_poses.py /workspace/sim-animate-environment/test/cut_roasted_beef/poses_bounds.npy /workspace/sim-animate-environment/test/cut_roasted_beef/cut_roasted_beef_ttt3r_refined.npy
```


```bash

nohup python3 -u orchestrator.py \
    --video_folder /workspace/sim-animate-environment/test/09_Alexa_Meade_Exhibit \
    --scene_name "09_Alexa_Meade_Exhibit" \
    --num_videos 7 \
    --end_frame 50 \
    --use_vggt > orchestrator_run.log 2>&1 &

```




```bash

nohup python3 -u orchestrator.py \
    --video_folder /workspace/sim-animate-environment/outputs/generated_scene_01 \
    --scene_name "generated_scene_01" \
    --num_videos 15 \
    --end_frame 50 \
    --v2v_model cosmos \
    --use_vggt > orchestrator_run.log 2>&1 &

```



```
pkill -f orchestrator.py; pkill -f preprocessor.py; pkill -f ttt3r_sfm.py && nohup python3 -u orchestrator.py \
    --image /workspace/sim-animate-environment/test/prototype_image_2.jpg \
    --prompt "Lightning strike in the scenery" \
    --scene_name "generated_scene_01" \
    --num_videos 20 \
    --end_frame 50 \
    --use_vggt \
    --v2v_model cosmos \
    > orchestrator_run.log 2>&1 &
```
