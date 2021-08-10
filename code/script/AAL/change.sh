for i in {1..10}
do
    sed -i "s/epoch 500/epoch 1000/g" train_baseline_AAL_lr1.e-3_dp0.6_fold${i}.sh
done
