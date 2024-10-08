## Conda environments

conda create -n remote_sensing_feature_dino python=3.10 anaconda -y

conda activate remote_sensing_feature_dino

## 
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install opencv-python tqdm torchmetrics tensorboard

pip install tensorboard


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
