export CUDA_VISIBLE_DEVICES="3"

# accelerate launch train_jempp.py --dataset_config configs/bloodmnist.yml --model_config configs/jempp_hparams.yml
python generate_tsne.py --dataset_config configs/bloodmnist.yml --model_config configs/jempp_hparams.yml