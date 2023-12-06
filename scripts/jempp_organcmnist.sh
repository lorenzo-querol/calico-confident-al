

export CUDA_VISIBLE_DEVICES="0,1,2,3"

accelerate launch train_jempp.py --dataset_config configs/organcmnist.yml --model_config configs/jempp_hparams.yml