# Salt-Image-Segmentation
- automatically and accurately identifies if a subsurface target is salt or not.

- Based on the dataset provided by a geoscience data company [seismic image](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data), 
which is a set of images are 101 x 101 pixels and each pixel is classified as either salt or sediment.

## Experiment I did:
### Experiment Name: CNN segmentation identifying salt 

- Files: [data_cleaning](https://github.com/LiuYuqing14/Salt-Image-Segmentation/blob/main/data_cleaning.py), [model_construction](https://github.com/LiuYuqing14/Salt-Image-Segmentation/blob/main/model_construct.py), and [data_evaluation](https://github.com/LiuYuqing14/Salt-Image-Segmentation/blob/main/model_evaluation.py)
- CNN model parameters:
  - optimizer = 'adam', Relu activation function
  - ResNet architecture and intersection-over-union (IoU) score evaluation
- Result:
  - 0.803 Public LB (0.812 Private LB)
  - <img alt="threshold" height="150" src="output_image/best%20thereshold.png" width="250"/>
  - <img alt="learing rate" height="150" src="output_image/learning%20rate.png" width="300"/>
- Reflection:
  - The incorrect labeling appears since there are limited training set and imbalanced set. We can not simply dividing set according to the percentage like the image below. The boundaries has very little percentage, and I infer there are catastrophic cancellation caused by computer during computation.
  - <img alt="reflect" height="300" src="output_image/reflect.jpg" width="300"/>
  
### Experiment Name: Which Encoder-Decoder Architecture workes best?

- files: 
- Model 1 (inspired by [Mr.ybabakhin's work](https://github.com/ybabakhin/kaggle_salt_bes_phalanx/tree/master))
  - input 101 -> pad to 512
  - Encoder:ResNet34
  - Decoder: conv3x3 + BN, Upsampling, scSE
- Model 2
  - input 101 -> resize to 512
  - Encoder: ResNet34
  - Decoder: conv3x3 + BN, Upsampling, scSE
  - (PS: I haven't run the entire models above because of time and technic limitation, but I tested partical of encoder part and combine with the conclusion from Mr.ybabakhin.)
- Model 3 (inspired by [this work](https://www.kaggle.com/code/meaninglesslives/getting-0-87-on-private-lb-using-kaggle-kernel/notebook))
  - 
- Result: 
