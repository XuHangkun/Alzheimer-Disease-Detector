for i in {1..10}
do
    mv train_baseline_Hammers_lr1.e-3_dp0.6_fold${i}.sh train_baseline_rBN_lr1.e-3_dp0.6_fold${i}.sh
    sed -i "s/Hammers/rBN/g" train_baseline_rBN_lr1.e-3_dp0.6_fold${i}.sh
done
