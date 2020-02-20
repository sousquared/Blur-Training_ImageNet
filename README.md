# BlurNet ImageNet Training

`logs/` will automaticaly be created.


## run examples
`python -u main_sou.py -a alexnet -b 256 --lr 0.01 --seed 42 --epochs 60 -n normal_60e_init-lr0.01_seed42 ../../../data/ImageNet/ILSVRC2012/ &`

`python -u main_blur-half.py -a alexnet -b 512 --lr 0.01 --seed 42 --epochs 60 -s 2 -k 13 13 -n blurhalf_s2_k13-13_b512 ../../../data/ImageNet/ILSVRC2012/ &`