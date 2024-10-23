## Conda environments

conda create -n remote_sensing_feature_dino python=3.10 anaconda -y

conda activate remote_sensing_feature_dino

## 
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install opencv-python tqdm torchmetrics tensorboard pytorch_warmup

pip install git+https://github.com/openai/CLIP.git

<!-- conda remove --name remote_sensing_feature_dino --all -->

tensorboard --logdir=classification/logs  --host "0.0.0.0"

on local pc

ssh -L 16006:127.0.0.1:6006 LStacchio@193.205.130.253

ssh -N -f -L localhost:16006:localhost:6006  LStacchio@193.205.130.253

docker commit c74e9c764ae5  lorenzostacchio/remotesensing:v1


## c++

### out of docker

### ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /root/anaconda3/envs/remote_sensing_feature_dino/lib/python3.10/site-packages/google/protobuf/pyext/_message.cpython-310-x86_64-linux-gnu.so)
apt-get update
apt-get install libstdc++6
conda install -c conda-forge libstdcxx-ng

### otherwise conda
conda install -c conda-forge gcc
conda install -c conda-forge gxx


## Datasets to explore
* https://github.com/google-research/google-research/blob/master/remote_sensing_representations/README.md#dataset-splits
* http://gpcv.whu.edu.cn/data/building_dataset.html
* http://gpcv.whu.edu.cn/data/
* https://github.com/satellite-image-deep-learning/datasets


### unrar staff

cd dir_rar/
unrar x -r NWPU-RESISC45.rar 

scp "C:\\Users\\Chiqu\\Downloads\\NWPU-RESISC45.rar" LStacchio@193.205.130.253:"'/disks/disk3/LorenzoStacchio/Remote Sensing/dataset/RECSIS45/'"

find . -type f | wc -l



### PYTHON
in data_loaders

wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py


## DATASETS

DIOR: https://drive.google.com/file/d/16JeLkqdOA1oF0WtyYdKnJhq3i9eEcX4_/view
