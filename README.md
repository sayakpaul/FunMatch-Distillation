# FunMatch-Distillation
TF2 implementation of knowledge distillation using the "function matching" hypothesis from the paper [Knowledge distillation:
A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237) by Beyer et al.

The techniques have been demonstrated using three datasets:
* [Pet37 dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
* [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* [Food101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

This repository provides [Kaggle Kernel notebooks](https://www.kaggle.com/kernels) so that we can leverage the _free_ TPu v3-8 to run
the long training schedules. Please refer to [this section](https://github.com/sayakpaul/FunMatch-Distillation#about-the-notebooks).

## Importance 

The importance of knowledge distillation lies in its practical usefulness. With the recipes from
"function matching", we  can now perform knowledge distillation using a principled approach
yielding student models that can actually match the performance of their teacher models. This essentially
allows us to compress bigger models into (much) smaller ones thereby reducing storage costs and
improving inference speed. 

## Key ingredients

* No use of ground-truth labels during distillation.
* Teacher and student should see same images during distillation as opposed to differently
  augmented views of same images.
* Aggressive form of [MixUp](https://arxiv.org/abs/1710.09412) as the key augmentation recipe. MixUp
  is paired with "Inception-style" cropping (implemented in [this script](https://github.com/sayakpaul/FunMatch-Distillation/blob/main/crop_resize.py)).
* A LONG training schedule for distillation. At least 1000 epochs to get good results _without_
  overfitting. The importance of a long training schedule is paramount as studied in the paper.
  
## Results

The table below summarizes the results of my experiments. In all cases, teacher is a BiT-ResNet101x3
model and student is a BiT-ResNet50x1. For fun, you can also try to distill into other model
families. BiT stands for "Big Transfer" and it was proposed in [this paper](https://arxiv.org/abs/1912.11370). 

|   Dataset  	|    Teacher/Student    	| Top-1 Acc on Test 	| Location 	|
|:----------:	|:---------------------:	|:-----------------:	|:--------:	|
| Flowers102 	|        Teacher        	|       98.18%      	|   [Link](https://bit.ly/2TER9tr)   	|
| Flowers102 	| Student (1000 epochs) 	|       81.02%       	|   [Link](https://git.io/JBO3Y)   	|
|    Pet37   	|        Teacher        	|       90.92%      	|   [Link](https://t.ly/hAKc)   	|
|    Pet37   	|  Student (300 epochs) 	|       81.3%       	|   [Link](https://git.io/JBO3i)   	|
|    Pet37   	| Student (1000 epochs) 	|        86%        	|   [Link](https://git.io/JBOsv)   	|
|   Food101  	|        Teacher        	|       85.52%      	|   [Link](https://bit.ly/3i7m9M0)   	|
|   Food101  	|  Student (100 epochs) 	|       76.06%        	|   [Link](https://git.io/JB3Xa)   	|

<sup>(**`Location` denotes the trained model location.**)</sup>

These results are consistent with Table 4 of the [original paper](https://arxiv.org/abs/2106.05237). 

_It should be noted that none of the above student training regimes showed signs of overfitting. Further
improvements can be done by training for longer._ The authors also showed that [Shampoo](https://github.com/google-research/google-research/tree/master/scalable_shampoo) can get to similar performance much quicker than Adam
during distillation. So, it may very well be possible to get this performance with fewer epochs
with Shampoo. 

A few differences from the original implementation:

* The authors use BiT-ResNet152x2 as a teacher. 
* The `mixup()` variant I used will produce a pair of duplicate images
  if the number of images is even. Now, for 8 workers it will become 8 pairs. 
  This may have led to the reduced performance. We can overcome this by using `tf.roll(images, 1, axis=0)` 
  instead of `tf.reverse` in the `mixup()` function. Thanks to Lucas Beyer for pointing this out.

## About the notebooks

All the notebooks are fully runnable on Kaggle Kernel. The only requirement is that you'd
need a billing enabled GCP account to use GCS Buckets to store data. 

|           Notebook          	|                                    Description                                   	| Kaggle Kernel 	|
|:---------------------------:	|:--------------------------------------------------------------------------------:	|:-------------:	|
|       `train_bit.ipynb`       	|                       Shows how to train the teacher model.                      	|      [Link](https://www.kaggle.com/spsayakpaul/train-bit)     	|
| `train_bit_keras_tuner.ipynb` 	| Shows how to run hyperparameter tuning using<br>Keras Tuner for the teacher model. 	|      [Link](https://www.kaggle.com/spsayakpaul/train-bit-keras-tuner)     	|
| `funmatch_distillation.ipynb` 	|         Shows an implementation of the recipes<br>from "function matching".         	|      [Link](https://www.kaggle.com/spsayakpaul/funmatch-distillation)     	|

These are only demonstrated on the Pet37 dataset but will work out-of-the-box for the other
datasets too. 

## TFRecords

For convenience, TFRecords of different datasets are provided:

|   Dataset  	| TFRecords 	|
|:----------:	|:---------:	|
| Flowers102 	|    [Link](https://git.io/JBOlw)   	|
|    Pet37   	|    [Link](https://git.io/JBOWr)   	|
|   Food101  	|    [Link](https://bit.ly/3iU0ZAq)   	|

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



