# DEUDF: Details Enhancement in Unsigned Distance Field Learning for High-fidelity 3D Surface Reconstruction (AAAI 2025)

We now release main code of our algorithm. 

## Install
```
    # install torch
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

    # install pymeshlab
    pip install pymeshlab
    
    # prepare for DCUDF
    pip install open3d trimesh matplotlib scipy scikit-image
    pip install PyMaxflow
```

## Prepare
We also provide the instructions for training your own data in the following.

### Data
First, you should put your own data to the `./data/3dscene` folder. The datasets is organised as follows:
```
│data/
│── 3dscene
│   ├── (dataname).ply
```
The default point cloud data format is `.ply`

### Run
To train your own data, simply run:
```
    # the default point cloud type is .ply
python net/classes/runner.py net/experiments/displacement_benchmark/ablation/ablation_phased_scaledTanh_yes_act_yes_baseLoss_yes_udf_esti.json --name $YOUR_POINT_CLOUD$
```

## Citation
```
@inproceedings{Xu2025DEUDF,
  title={Details Enhancement in Unsigned Distance Field Learning for High-fidelity 3D Surface Reconstruction},
  author = {Cheng Xu, Fei Hou, Wencheng Wang, Hong Qin\textsuperscript, Zhebin Zhang, Ying He}
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
