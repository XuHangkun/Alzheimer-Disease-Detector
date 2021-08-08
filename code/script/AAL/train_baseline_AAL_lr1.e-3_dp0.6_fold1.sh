#!/bin/bash
######## Part 1 #########
# Script parameters     #
#########################

# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu

# Specify the QOS, mandatory option
#SBATCH --qos=normal

# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=junogpu

# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=Baseline

# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/hw_ad_competition/log/baseline-AAL-%j.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=32GB

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1

######## Part 2 ######
# Script workload    #
######################

############ TRAIN ################
cd /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/hw_ad_competition/code
python train.py --lr 1.e-3 --epoch 500 --atlas AAL --model_name splitbaseline \
--train_info_path ../model/train_AAL_lr1.e-3_dp0.6_fold1_info.csv \
--valid_info_path ../model/valid_AAL_lr1.e-3_dp0.6_fold1_info.csv \
--model_path ../model/baseline/baseline_AAL_lr1.e-3_dp0.6_fold1_epoch%d.pth \
--dropout 0.6 --save_start_epoch 100 --kth_fold 1
