# THACIL Network
This repo is our implementation for paper: Temporal Hierarchical Attention at Category- and Item-Level for Micro-Video Click-Through Prediction

**Please cite our MM'18 paper if you use our codes. Thanks!**
```
@inproceedings{chen2018thacil,
  title={Temporal Hierarchical Attention at Category- and Item-Level for Micro-Video Click-Through Prediction},
  author={Chen, Xusong and Liu, Dong and Zha, Zheng-Jun and Zhou, Wengang and Xiong, Zhiwei and Li, Yan},
  booktitle={MM},
  year={2018}
}
```

## Environment Setting
- Python 3.6 (Anaconda3)
- Tensorflow 1.4

## Usage
### Download dataset and preprocess
* Step 1: Download our dataset, named [MicroVideo-1.7M](https://pan.baidu.com/s/1iQO23zSXPv1b9y2uLxaJmg) (password: 4xsg), which has 12,737,619 interactions that 10,986 users have made on 1,704,880 micro-videos. And move it to `data/input/` folder.
* Step 2: Generate intermediate file for training, and move it to `data/input/` folder.
    - Generate `user_click_ids.npy`:
        ```
        python generate_data.py --train-data-path ../../data/input/train_data.csv --save-path ../../data/input/
        ```
### Training and Evaluation
- Training
```
cd src
python launcher.py  --phase train
```
- Evaluation
```
cd src
python launcher.py  --phase test
```

## Acknowledgements
Some implementations consult the [ATRank](https://github.com/jinze1994/ATRank), and [DiSAN](https://github.com/taoshen58/DiSAN).

