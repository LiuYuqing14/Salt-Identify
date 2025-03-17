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
  - <img alt="threshold" height="300" src="output_image/best%20thereshold.png" width="500"/>
  - <img alt="learing rate" height="300" src="output_image/learning%20rate.png" width="600"/>
- Reflection:
  - The incorrect labeling appears since there are limited training set and imbalanced set. We can not simply dividing set according to the percentage like the image below. The boundaries has very little percentage, and I infer there are catastrophic cancellation caused by computer during computation.
  - <img alt="reflect" height="300" src="output_image/reflect.jpg" width="600"/>
  
### Experiment Name: Which Encoder-Decoder Architecture workes best?

- files: 
