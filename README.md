# Burned-Land-Detection

Repo containing code from research on Convolution Neural Networks on detection of cover and type of burned land areas in the Amazon Rainforest.

In this project, a model based on the U-net architecture was trained for semantic segmentation. From 2 images of an area, one prior and one post some events of fire, we wish to generate a mask with 3 classes: Non burned areas (black pixels on the mask), Burned areas that were from a forest area prior to the fire (dark green pixels on the mask) and burned areas that were pasture prior to the fire (light green pixels on the mask)   

The following table shows the metrics of the model for each class:

| Class      | Accuracy | Precision | Recall   | F1       |
|------------|----------|-----------|----------|----------|
| Not Burned | 0.931086 | 0.976243  | 0.934025 | 0.954668 |
| Forest     | 0.984059 | 0.842559  | 0.740085 | 0.788004 |
| Pasture    | 0.925765 | 0.745628  | 0.902305 | 0.816518 |

Example of input and output expected from the neural net:

![Screenshot from 2021-09-30 20-16-24](https://user-images.githubusercontent.com/63306096/135854501-fbef85e0-a132-49a5-ac83-fb0b2ea3d12c.png)
|:--:| 
| *Input: Image of an area prior to a set of fires* |

|![Screenshot from 2021-09-30 20-15-59](https://user-images.githubusercontent.com/63306096/135854503-bfaf710f-22ec-433d-83e5-ab89b1c7fbf5.png)|
|:--:| 
| *Input: Image of the same area as the previous figure, post a set of fires* |

![prediction](https://user-images.githubusercontent.com/63306096/135854424-e195cda2-551c-4426-9764-5ba75193be44.png)
|:--:| 
| *Output: Mask generated from both images above, representing the extent and origin of all burned lands presented in the image post the set of fires.* |
