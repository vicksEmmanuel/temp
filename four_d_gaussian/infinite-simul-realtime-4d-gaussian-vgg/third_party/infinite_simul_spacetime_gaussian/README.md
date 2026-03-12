### Installation

```sh
chmod +x ./scripts/setup.sh
./scripts/setup.sh
```


### Data Setup

```sh
export QT_QPA_PLATFORM=offscreen
xvfb-run python3 setup_data.py
```


### Train

```
python3 train.py --model_path log/cooking_beef_3 --source_path /home/featurize/work/infinite-simul-4d-gaussian/data/cut_roasted_beef_3/colmap_10 --config /home/featurize/work/infinite-simul-4d-gaussian/configs/general.json
```

```
python3 train.py --model_path log/cooking_beef_2 --source_path /home/featurize/work/infinite-simul-4d-gaussian/data/cut_roasted_beef_2/colmap_10 --config /home/featurize/work/infinite-simul-4d-gaussian/configs/general.json
```

