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





```bash

nohup python3 -u orchestrator.py \
    --video_folder /workspace/sim-animate-environment/outputs/coffee_martini \
    --scene_name "coffee_martini" \
    --num_videos 20 \
    --end_frame 300 \
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
