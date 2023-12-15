for i in {1..5}
do  
    accelerate launch train_jempp.py --dataset_config configs/bloodmnist.yml --model_config configs/jempp_hparams.yml --seed $i
done