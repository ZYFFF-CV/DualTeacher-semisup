# DualTeacher-semisup
An official implementation of "Dual teacher: Improving the reliability of pseudo labels for semi-supervised oriented object detection"
<!-- A semi-supervised learning method, build on MMRotate -->
## TODO
Trainin code and config will be updated after the paper is accepted
## Installation
1. Install the key packages
- `Anaconda3` with `python=3.8`
- `Pytorch=1.9.0`
- `mmdetection=2.25.1`
- `mmroatate=0.3.2`
- `mmcv=1.6.0`

2. Clone the repository
```shell script
git clone https://github.com/ZYFFF-CV/DualTeacher-semisup.git

```
3. Install requirement packages
```
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install this package
```shell script
cd DualTeacher-semisup
python -m pip install -e .
```

```
@article{fang2024dual,
  title={Dual teacher: Improving the reliability of pseudo labels for semi-supervised oriented object detection},
  author={Fang, Zhenyu and Ren, Jinchang and Zheng, Jiangbin and Chen, Rongjun and Zhao, Huimin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
5. Remove unnecessary packages
```shell script
rm -r ssod.egg-info
```

