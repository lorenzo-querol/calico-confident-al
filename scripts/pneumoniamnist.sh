export CUDA_VISIBLE_DEVICES="2,3,4,5"

accelerate launch train_jempp.py --dataset_config configs/medmnist/pneumoniamnist.yml --model_config configs/jempp_hparams.yml