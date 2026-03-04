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