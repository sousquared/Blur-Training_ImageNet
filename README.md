# BlurNet ImageNet Training

`logs/` will automaticaly be created.


## Preparation
Install Python Packages  
```bash
$ pip install -r requirements.txt
```
Or pull and run [docker image][docker-blurnet] (e.g. blurnet:1.0) I made for this experiments.  


## run examples
<<<<<<< HEAD
General usage example:
```bash
$ cd ./blurnet
$ python main.py --mode [TRAINING MODE] -n [EXPERIMENT NAME] [IMAGENET_PATH]
```  

For `main.py`, you need to use `--exp-name` or `-n` option to define your experiment's name. Then the experiment's name is used for managing results under `logs/` directory.   
You can choose the training mode from {normal,blur-all,blur-half,blur-step,blur-half-data} by using `--mode [TRAINING MODE]` option.

- **normal**  
This mode trains Normal alexnetCifar10.  
usage example:  
```bash
$ python -u main.py --mode normal -a alexnet -b 256 --lr 0.01 --seed 42 --epochs 60 -n normal_60e_init-lr0.01_seed42 /mnt/data/ImageNet/ILSVRC2012/ &
```

- **blur-all**  
This mode blurs ALL images in the training mode.  
usage exmaple:  
```bash
$ python -u main.py --mode blur-all -a alexnet --seed 42 --lr 0.01 -s 3 --epochs 60 -b 512 -n blur-all_s3 /mnt/data/ImageNet/ILSVRC2012/
```

- **blur-half**    
This mode blurs first half epochs (e.g. first 30 epochs in 60 entire epochs) in the training.
usage example:  
```bash
$ python -u main.py --mode blur-half -a alexnet --seed 42 --lr 0.01 -s 3 --epochs 60 -b 512 -n blur-half_s3 /mnt/data/ImageNet/ILSVRC2012/
```

- **blur-half-data**    
This mode blurs half training data.
usage example:  
```bash
$ python -u main.py --mode blur-half-data -a alexnet --seed 42 --lr 0.01 -s 3 --epochs 60 -b 512 -n blur-half-data_s3 /mnt/data/ImageNet/ILSVRC2012/
```

- **blur-step**  
This mode blurs images step by step (e.g. every 10 epochs).  
usage example:  
```bash
$ python -u main.py --mode blur-step -a alexnet --seed 42 --lr 0.01 --epochs 60 -b 512 -n blur-step /mnt/data/ImageNet/ILSVRC2012/
```

- `--blur-val`   
This option blurs validation data as well. 
usage example:  
```bash
$ python -u main.py --mode blur-half-data --blur-val -a alexnet --seed 42 --lr 0.01 -s 3 --epochs 60 -b 512 -n blur-half-data_blur-val_s3 /mnt/data/ImageNet/ILSVRC2012/
```

- `--resume [PATH TO SAVED MODEL]`   
This option trains Normal alexnetCifar10 from your saved model.  
usage example:  
```bash
$ python -u main.py --mode blur-half-data -a alexnet --seed 42 --lr 0.01 -s 3 --epochs 60 -b 512 --resume ../logs/models/blur-half_s1/model_060.pth.tar -n blur-half-data_s3_from60e /mnt/data/ImageNet/ILSVRC2012/
```

## citation
Training scripts and functions are strongly rely on [pytorch tutorial][pytorch-tutorial] and [pytorch imagenet trainning example][pytorch-imagenet].



[pytorch-tutorial]:https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
[pytorch-imagenet]:https://github.com/pytorch/examples/blob/master/imagenet/main.py
[docker-blurnet]:https://hub.docker.com/r/sousquared/blurnet