uv run python multicondiffusion.py \
  --prompt_file examples/29_real_campus_3.txt \
  --input_image /mnt/c/Users/kodeb/OneDrive/Desktop/vicks/Infinite-Simul/sim-animate-environment-2/test/environment1/images/6.jpg \
  --output_dir output \
  --steps 10 \
  --iterations 1 \
  --debug

  07076272258



python multicondiffusion.py \
  --prompt_file examples/29_real_campus_3.txt \
  --input_image /mnt/c/Users/kodeb/OneDrive/Desktop/vicks/Infinite-Simul/sim-animate-environment-2/test/environment1/images/tetter.png \
  --output_dir output_wide \
  --W 2560 \
  --iterations 5


python multicondiffusion_panorama.py \
  --prompt_file examples/29_real_campus_3.txt \
  --input_image /mnt/c/Users/kodeb/OneDrive/Desktop/vicks/Infinite-Simul/sim-animate-environment-2/test/environment1/images/tetter.png \
  --output_dir output_360_2 \
  --W 3912 \
  --iterations 3


python depth_estimation.py \
  --input_image output_360/29_real_campus_3_seed0/final_output_29_real_campus_3.png \
  --output_dir output_depth \
  --mode wide


python ldi_generation.py \
  --input_image output/29_real_campus_3_seed0/final_output_29_real_campus_3.png \
  --input_depth output_depth/depth.npy \
  --output_dir output_ldi \
  --num_layers 4



python train_gsplat.py \
  --ldi_dir output_ldi \
  --output scene_optimized.ply \
  --num_iterations 300 \
  --num_views 16


  export KMP_DUPLICATE_LIB_OK=TRUE && uv run python generate_poses_da3.py --source test/environment1 --output poses_bounds_da3.npy --checkpoints /Users/victorumesiobi/Desktop/vicks/Infinite_Simul/sim-animate-environment/checkpoints