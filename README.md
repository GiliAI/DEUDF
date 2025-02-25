## The source code is coming soon
The path of 3D model is edited in .json file\
Default filetype is .ply
```
python net/classes/runner.py net/experiments/displacement_benchmark/ablation/ablation_phased_scaledTanh_yes_act_yes_baseLoss_yes_udf_esti.json --name your_model
```

# DEUDF: Details Enhancement in Unsigned Distance Field Learning for High-fidelity 3D Surface Reconstruction (AAAI 2025)

We now release main code of our algorithm. 

## Install

## Prepare
We also provide the instructions for training your own data in the following.

### Data
First, you should put your own data to the `./data/input` folder. The datasets is organised as follows:
```
│data/
│── input
│   ├── (dataname).ply
```
The default point cloud data format is `.ply`

### Run
To train your own data, simply run:
```
python run.py 
```


