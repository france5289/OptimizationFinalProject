# Optimization Final Project
Ranger optimzer v.s other optimizer (SGD and Adam) on ResNet-18 with CIFAR-10 dataset

## Ranger optimizer
> Ranger - a synergistic optimizer combining RAdam (Rectified Adam) and LookAhead, and now GC (gradient centralization) in one optimizer. 
- [Ranger Official Github Page](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
- [RAdam Paper Link](https://arxiv.org/pdf/1908.03265.pdf)
- [Lookahed Paper Link](https://arxiv.org/pdf/1907.08610v1.pdf)

## Environment
- OS : Ubuntu 18.04
- CUDA version : 10.2
- PyTorch : 1.5.0
- GPU : RTX 2070 Super
- CPU : AMD Ryzen R7-3700X

## Usage
### Environmental setup
> It's recommended to use a python virtual environments 

Clone this repo and install all dependencies
```bash
git clone https://github.com/france5289/OptimizationFinalProject.git
cd OptimizationFinalProject
pip3 install -r requirements.txt
```
After this you should install *Ranger* from its official Github page  
[Ranger Official Github Page](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)  
### How to run your experiments
#### Training
there are three files to train ResNet-18 model with different optimizers  
- `train_SGD.py`
  - train ResNet-18 with SGD optimzier
- `train_Adam.py`
  - train ResNet-18 with Adam optimizer
- `train_Ranger.py`
  - train ResNet-18 with Ranger optimizer  

All of these files mentioned above will read a JSON file which records relative hyperparameters. So you shoud modify the following three JSON file before running your experiments  
- `hyperparameters_SGD.json`
- `hyperparameters_Adam.json`
- `hyperparameters_Ranger.json`   

> Note : if you want to add more hyperparameters, you sould not only modify relative JSON file but also modify relative code of `config` object (in config/Resnet18_config.py)

Finally type the following command(here i use SGD for example) :   
```bash
python3 train_SGD.py
```
#### Inference
Type the following commands : 
```bash
python3 inference.py
```
Then type in the filename of your checkpoint of model, it will print testing accuray of your model. 
> Note : model checkpoint will stored under `/Cifar10_model` by default

### Tensorboard
All tensorboard event files will stored under `/exp_log` by default. The filename will be equal to the `expname` filed of JSON file(`hyperparameters_{optimizer}.json`)  
> Note : You should not use the same `expname` of every experiments, or you will find that scalar of tensorboard will be overlapping
## Experimental Result
- Model : ResNet-18
- Dataset : CIFAR-10
- batch size : 128
- epochs : 200
- lrate = { 0.1, 0.03, 0.01, 0.003 }
- SGD with momentum : 0.9

|   Expname   | lrate | min_train_loss     | Test Acc  |
|:-----------:|:-----:|:------------------:|:---------:|
|  SGD_exp1   |  0.1  |     **0.0319**     |   79.98   |
|  SGD_exp2   | 0.01  |     0.0697         |   76.0    |
|  SGD_exp3   | 0.03  |     0.0346         |   77.79   |
|  SGD_exp4   | 0.003 |     0.2235         |   73.45   |
|  Adam_exp1  |  0.1  |     0.3428         |   76.92   |
|  Adam_exp2  | 0.01  |     0.0604         |   82.83   |
|  Adam_exp3  | 0.03  |     0.0888         |   80.81   |
|  Adam_exp4  | 0.003 |     0.051          |   82.46   |
| Ranger_exp1 |  0.1  |     0.0712         |   82.71   |
| Ranger_exp2 | 0.01  |     0.0513         |   83.93   |
| Ranger_exp3 | 0.03  |     0.0563         | **83.96** |
| Ranger_exp4 | 0.003 |     0.0476         | **83.96** |