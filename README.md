# Salt-Image-Segmentation
- automatically and accurately identifies if a subsurface target is salt or not.

- Based on the dataset provided by a geoscience data company [seismic image](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data), 
which is a set of images are 101 x 101 pixels and each pixel is classified as either salt or sediment.

## Experiment I did:
### Experiment Name: CNN segmentation identifying salt 

- files: [data_cleaning](https://github.com/LiuYuqing14/Salt-Image-Segmentation/blob/main/data_cleaning.py), [model_construction](https://github.com/LiuYuqing14/Salt-Image-Segmentation/blob/main/model_construct.py), and [data_evaluation](https://github.com/LiuYuqing14/Salt-Image-Segmentation/blob/main/model_evaluation.py)
- CNN model parameters:
  - optimizer = 'adam', Relu activation function
  - ResNet architecture and intersection-over-union (IoU) score evaluation
- result:
  - <img alt="threshold" height="1256" src="output_image/best%20thereshold.png" width="1914"/>