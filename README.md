# DualTeacher-semisup
An official implementation of "SOOD: Towards Semi-Supervised Oriented Object Detection"
<!-- A semi-supervised learning method, build on MMRotate -->
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

5. Remove unnecessary packages
```shell script
rm -r ssod.egg-info
```

