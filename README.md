# Monodepth2

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

If you find our work useful in your research please consider citing our paper:

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```



## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
We ran our experiments with PyTorch 0.4.1, CUDA 9.1, Python 3.6.6 and Ubuntu 18.04.
We have also successfully trained models with PyTorch 1.0, and our code is compatible with Python 2.7. You may have issues installing OpenCV version 3.3.1 if you use Python 3.7, we recommend to create a virtual environment with Python 3.6.6 `conda create -n monodepth2 python=3.6.6 anaconda `.

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts.

We also recommend using `pillow-simd` instead of `pillow` for faster image preprocessing in the dataloaders. -->


## 학습 데이터 준비

- 데이터 셋 다운로드 
아래 코드를 이 폴더에서 실행해서 데이터셋을 다운 받으실 수 있습니다.:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```

**Warning:** 데이터가 총 175GB이니 용량을 유의하시길 바랍니다. 


## ⏳ Training

- 실행 명령어 입니다. 위에서 kitti_data를 다운 받았다면 아래 코드를 실행시켜서 학습시켜 주면 됩니다. 
    - log가 저장되는 default가 ~/tmp 이며 변경하고 싶으면 log_directory 를 원하는 경로로 변경바랍니다. 
    - log_name 또한 변경하여 실해시켜주시길 바랍니다. 
    
**Monocular training:**
```shell
python train.py --model_name {log_name} --png --log_dir {log_directory}
```



### GPUs

- 현재 코드는 단일 GPU로 동작할 수 있게 되어있습니다. 만약 GPU 번호를 바꾸고 싶으시면 아래와 같이 `CUDA_VISIBLE_DEVICES`의 번호를 변경해주시길 바랍니다.

```shell
CUDA_VISIBLE_DEVICES=2 python train.py --model_name mono_model
```

모든 실험은 Titan Xp 에서 실험했으며 아래와 같이 GPU를 사용하게 됩니다. 

| Training modality | Approximate GPU memory  | Approximate training time   |
|-------------------|-------------------------|-----------------------------|
| Mono              | 9GB                     | 12 hours                    |



### 💽 Finetuning a pretrained model

- 학습해둔 모델을 finetuning 하고싶으시면 아래와 같이 --load_weights_folder를 이용해 경로를 지정해주시길 바랍니다. :
```shell
python train.py --model_name finetuned_mono --load_weights_folder ~/tmp/mono_model/models/weights_19
```


### 🔧 Other training options

- 다른 학습 옵션을 보고싶으시면 ```option.py'''를 보시길 바랍니다 .

## 평가 영상 생성
- 리더보드에 제출하기 위한 영상을 생성하기 위해서는 아래와 같이 실행시켜주시길 바랍니다. 
    -학습하신 모델의 경로를 load_weights_folder에 넣고 돌리시면 현재폴더에 ```disp_eigen_split.npz``` 가 생성될 것입니다. 그 파일을 제출해주시면 됩니다. 
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/stereo_model/models/weights_19/ --eval_mono
```

## 👩‍⚖️ License
Copyright © Niantic, Inc. 2019. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
