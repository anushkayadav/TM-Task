# Appraoch 2

## Dataset Used 
  - Scraped images from google using Javascript and Python Scripts
  - Scraped about 1000 images 
  - This was a more diverse data as different garments of each category were used rather than using very similar images per class as in our previous Dataset.
  
## Classes

For comparision purpose used the same classes as in first approach i.e Saree , Kurti and Shirt.

## Preprocessing

- Image Augmentation techniques were applied.
- Data was normalised and converted into tensor before passing into the model


## Model

**ResNet-50** 

Training the last 2 layers i.e layer 4 and fc gave better results than training just the last layer. So 2 layer training was considered. 

 - Frozen Layers : conv1 , bn1 , relu, maxpool, layer1, layer2, layer3, avgpool
 - Unfrozen Layers :  fc. layer4

## Results
```
Test Loss: 0.393063


Test Accuracy: 83% (137/165)
```
![res11.png](/images/res11.PNG) 
![res12.png](/images/res12.PNG) 
![res13.png](/images/res13.PNG) 


##  Occlusion Heat Map

