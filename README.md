# FunMatch-Distillation
TF2 implementation of knowledge distillation using the "function matching" hypothesis from the paper [Knowledge distillation:
A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237) by Beyer et al.

The techniques have been demonstrated using the [Pet37 dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/). 

This repository provides Kaggle Kernels notebooks so that we can leverage the _free_ TPu v3-8 to run
the long training schedules. 

## Importance 

The importance of this technique lies in its practical usefulness. With the recipes from
"function matching", we  can now perform knowledge distillation using a principled approach
yielding student models that can actually match the performance of their teacher models. 

This essentially allows us to compress bigger models into (much) smaller ones thereby reducing 
storage costs and improving inference speed. 

## Key ingredients

* No use of ground-truth labels during distillation.
* Teacher and student should see same images during distillation as opposed to differently
  augmented views of same images.
* Aggressive form of [MixUp](https://arxiv.org/abs/1710.09412) as the key augmentation recipe. MixUp
  is paired with "Inception-style" cropping.
* A LONG training schedule for distillation. At least 1000 epochs to get good results _without_
  overfitting. The importance of a long training schedule is paramount as studied in the paper.
  
## Results

First, a well-performing teacher model is generated. In this case, it's a BiT-ResNet101x3 as 
opposed to a BiT-ResNet152x2 (used in the original paper). This model yields 90.93% top-1 accuracy
on the test set of Pet37. The authors distill into a BiT-ResNet50 for different training regimes:
300 epochs, 1000 epochs, 3000 epochs, etc. 

I have only run the experiments for 300 epochs and 1000 epochs. 

| **Epochs** 	| **Top-1 Acc** 	|
|:------:	|:---------:	|
|   300  	|   81.3%   	|
|  1000  	|   86%        	|

_It should be noted that none of the above training regimes showed signs of overfitting._

## About the notebooks

All the notebooks are fully runnable on Kaggle Kernel. The only requirement is that you'd
need a billing enabled GCP account to use GCS Buckets to store data. 

|           Notebook          	|                                    Description                                   	| Kaggle Kernel 	|
|:---------------------------:	|:--------------------------------------------------------------------------------:	|:-------------:	|
|       `train_bit.ipynb`       	|                       Shows how to train the teacher model.                      	|      [Link](https://www.kaggle.com/spsayakpaul/train-bit)     	|
| `train_bit_keras_tuner.ipynb` 	| Shows how to run hyperparameter tuning using<br>Keras Tuner for the teacher model. 	|      [Link](https://www.kaggle.com/spsayakpaul/train-bit-keras-tuner)     	|
| `funmatch_distillation.ipynb` 	|         Shows an implementation of the recipes<br>from "function matching".         	|      [Link](https://www.kaggle.com/spsayakpaul/funmatch-distillation)     	|

## TFRecords and pre-trained weights

For reproducibility, TFRecords and pre-trained model weights are provided:

* [TFRecords](https://github.com/sayakpaul/FunMatch-Distillation/releases/download/v1.0.0/tfrecords_pets37.tar.gz)
* [Teacher model (BiT-ResNet101x3)](https://www.kaggle.com/spsayakpaul/bitresnet101x3-pet37)
* [Student model (BiT-ResNet50)](https://github.com/sayakpaul/FunMatch-Distillation/releases/download/v2.0.0/S-r50x1-128-300.tar.gz) (300 epochs)
* [Student model (BiT-ResNet50)](https://github.com/sayakpaul/FunMatch-Distillation/releases/download/v2.0.0/S-r50x1-128-1000.tar.gz) (1000 epochs)

## Paper citation

```
@misc{beyer2021knowledge,
      title={Knowledge distillation: A good teacher is patient and consistent}, 
      author={Lucas Beyer and Xiaohua Zhai and Amélie Royer and Larisa Markeeva and Rohan Anil and Alexander Kolesnikov},
      year={2021},
      eprint={2106.05237},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

Huge thanks to [Lucas Beyer](https://scholar.google.com/citations?user=p2gwhK4AAAAJ&hl=en) 
(first author of the paper) for providing suggestions on the initial version of the implementation.

Thanks to the [ML-GDE program](https://developers.google.com/programs/experts/) for providing GCP credits.

Thanks to [TRC](https://sites.research.google/trc/) for providing Cloud TPU access. 



