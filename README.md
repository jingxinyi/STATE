# STATE: Learning Structure and Texture Representations for Novel View Synthesis
This repository contains a pytorch implementation of [STATE: Learning Structure and Texture Representations for Novel View Synthesis](https://cic.tju.edu.cn/faculty/likun/projects/STATE/assets/STATE_final.pdf), CVMJ 2022. 
 
 [Project Page](https://cic.tju.edu.cn/faculty/likun/projects/STATE/index.html)  
 [Supp.](https://cic.tju.edu.cn/faculty/likun/projects/STATE/assets/supp.pdf)

## Requirement

```
conda env create -f environment.yaml
```

## Data
The datasets of Car and Chair can be downloaded from [TBN](https://github.com/kyleolsz/TB-Networks).

## Train
```
python train_car.py -c config_car.json [-r path_to_checkpoint]
```

## Test
```
python test.py -c config_car.json -r path_to_checkpoint
```

## Citation
```
@inproceedings{STATE,
  author = {Xinyi Jing and Qiao Feng and Yu-kun Lai and Jinsong Zhang and Yuanqiang Yu and Kun Li},
  title = {STATE: Learning structure and texture representations for novel view synthesis},
  booktitle = {Computational Visual Media},
  year={2022},
}
```
