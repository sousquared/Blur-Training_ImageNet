# BlurNet ImageNet Training

`logs/` will automaticaly be created.


## Preparation
Install Python Packages  
```bash
$ pip install -r requirements.txt
```
Or pull and run [docker image][docker-blurnet] (e.g. blurnet:1.0) I made for this experiments.  


## run examples
`python -u main_sou.py -a alexnet -b 256 --lr 0.01 --seed 42 --epochs 60 -n normal_60e_init-lr0.01_seed42 ../../../data/ImageNet/ILSVRC2012/ &`

`python -u main_blur-half.py -a alexnet -b 512 --lr 0.01 --seed 42 --epochs 60 -s 2 -k 13 13 -n blurhalf_s2_k13-13_b512 ../../../data/ImageNet/ILSVRC2012/ &`

`python -u main.py -a alexnet --seed 42 --lr 0.01 --mode blur-half -s 3 -k 19 19 --epochs 60 -b 512 -n blur-half_s3_k19-19_b512 /mnt/data/ImageNet/ILSVRC2012/`

`python -u main.py -a alexnet --seed 42 --lr 0.01 --mode blur-half-data -s 3 --epochs 60 -b 512 -n blur-half-data_s3_b512 /mnt/data/ImageNet/ILSVRC2012/`

`python -u main.py -a alexnet --seed 42 --lr 0.01 --mode blur-step --epochs 60 -b 512 -n blur-step_b512 /mnt/data/ImageNet/ILSVRC2012/`


## citation
Training scripts and functions are strongly rely on [pytorch tutorial][pytorch-tutorial] and [pytorch imagenet trainning example][pytorch-imagenet].



[pytorch-tutorial]:https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
[pytorch-imagenet]:https://github.com/pytorch/examples/blob/master/imagenet/main.py
[docker-blurnet]:https://hub.docker.com/r/sousquared/blurnet