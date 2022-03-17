# Install
In order to run the scripts, please follow the steps below;
1. Please download and install the implementation of [ANS & Occupancy]({https://github.com/facebookresearch/OccupancyAnticipation]) as required.
2. Replace `ans.py`, `policy.py` and `occant_exp_trainer.py` in `occant_baselines/rl` with `ans.py`, `policy.py` and `occant_exp_trainer.py` in `./SANS` of our directory.
3. Copy `./SANS/config/train/*yaml` and `./SANS/config/test/*yaml` to `configs/model_configs/ans_depth`.

# Training
1. To train ANS, please run 
```
python run.py --exp-config configs/model_configs/ans_depth/ppo_exploration_cat_allolocal_alloglobal_cart_ansnet.yaml --run-type train
```
2. To train S-ANS, please run 
```
python run.py --exp-config configs/model_configs/ans_depth/ppo_exploration_allolocal_alloglobal_cart_ansnetexact_shareconv_rllocal__actorBP_criticBPGPP.yaml --run-type train
```
# Testing
1. To evaluate ANS on Gibson, please run
```
python run.py --exp-config configs/model_configs/ans_depth/ppo_exploration_myeval_novideo_cat_allolocal_alloglobal_cart_ansnet.yaml --run-type eval
```
2. To evaluate ANS on MP3D, please run
python run.py --exp-config configs/model_configs/ans_depth/ppo_exploration_myeval_mp3d_novideo_cat_allolocal_alloglobal_cart_ansnet.yaml --run-type eval
3. To evaluate S-ANS on Gibson, please run
```
python run.py --exp-config configs/model_configs/ans_depth/ppo_exploration_myeval_video_allolocal_alloglobal_cart_ansnetexact_shareconv_rllocal__actorBP_criticBPGPP.yaml --run-type eval
```
4. To evaluate S-ANS on MP3D, please run
```
python run.py --exp-config configs/model_configs/ans_depth/ppo_exploration_myeval_mp3d_video_allolocal_alloglobal_cart_ansnetexact_shareconv_rllocal__actorBP_criticBPGPP.yaml --run-type eval
```