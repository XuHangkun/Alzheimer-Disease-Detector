for i in {1..10}
do
    sed -i "s/fold1/fold${i}/g" train_baseline_Hammers_lr1.e-3_dp0.6_fold${i}.sh
    sed -i "s/fold 1/fold ${i}/g" train_baseline_Hammers_lr1.e-3_dp0.6_fold${i}.sh
done
