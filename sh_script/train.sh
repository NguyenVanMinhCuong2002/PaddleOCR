# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/rec_svtrnet.yml