# FunMatch-PyTorch
PyTorch implementation of [FunMatch](https://arxiv.org/abs/2106.05237)

For convenience, some of the code is out of argument. You can obtain the weight from [Big Transfer](https://github.com/google-research/big_transfer)

Reference:
1. https://github.com/sayakpaul/FunMatch-Distillation/tree/main
2. https://github.com/pytorch/vision/tree/main/references/classification
3. https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

## Progress
- [24/03/03] PyTorch implementation

- [24/03/17] 
    * It's the MOST significant method across all knowledge distillation methods that I've tried before, It makes a great improvement in my other projects. I will try to do comprehensive research on it.
    * I also tried torch.compile but encountered some error when use torchrun/distributed.launch. It's faster but still not feasible so far.
    * I noticed that tensor flag is False after PyTorch 1.12 and set it as True in the code.
    * It should be applied in every image classification, and I would like to write a general one.


### TODO

|   Dataset  	|    Teacher/Student    	| Top-1 Acc on Test 	|
|:----------:	|:---------------------:	|:-----------------:	|
| [Pet37](http://www.robots.ox.ac.uk/~vgg/data/pets/)   	        | BiT-ResNet1521x2        	   |       N/A      	| 
| [Pet37](http://www.robots.ox.ac.uk/~vgg/data/pets/)           	| BiT-ResNet50x1 (300 epochs)  |       N/A      	| 
| [Pet37](http://www.robots.ox.ac.uk/~vgg/data/pets/)   	        | BiT-ResNet50x1 (1000 epochs) |       N/A      	| 
| [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) 	| BiT-ResNet152x2              |       N/A      	| 
| [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) 	| BiT-ResNet50x1 (1000 epochs) |       N/A      	| 
| [Food101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) | BiT-ResNet152x2        	   |       N/A      	| 
| [Food101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) | BiT-ResNet50x1 (1000 epochs) |       N/A      	| 

- [x] Add requirements
- [x] Align the procedure to the TF version
- [ ] Reproduce a similar result on a dataset
- [ ] Revision
- [ ] Add Shampoo
- [ ] Quantization
- [ ] Other researches

## Instructions
```
torchrun --nproc_per_node=[GPU_NUM] train.py --epochs 300 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --data-path <DATA_PATH>
```
for distributed training or 
```
python3 train.py --epochs 300 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --data-path <DATA_PATH>
```
for single GPU