#  Product Identification using Deep Learning  

**Input To WebApp** : Upload Image from Local Machine and Submit
![pg1.png](/images/pg1.png) 

**Result On WebApp** : Product Name is dispalyed in the highlighted section.
![pg2.png](/images/pg2.png) 


## Dataset Used : 
The Provided dataset had 20 images in total.
10 images of kurti , 4 images of shirt and 6 images of saree.   
[Data Set Provided](/images) 

## Classes : 
I have considered 3 classes - 
  - Kurti
  - Saree
  - Shirt


## Challenges Faced
**Challenge 1** : Very small dataset leading to lack of generalization.

        -Solution : Using different data augmentation technique using ImgAug.
 **Challenge 2** : All T-shirt images consisted of black colored Tshirt and augmentations such as gamma contrast, hue saturation, linear contrast etc. don't have much effect on black color.

        -Solution : As the model was not generalizing on colors of shirts, This problem was solved using inversion augmentation along with other techniques.


## Preprocessing 

1. Dta augmentations can help in increase the training examples on the fly during training and help to increase the performance of architechture due to randomly indtroduced variations of the sampled examples which i have implemented here using transforms

Used ImgAug to apply different transformations on the images.

    - Horizontal and Vertical Flip
    - Invert
    - Crop and Pad
    - Affine transformations
    - Blur and Emboss
    - Changing brightness and contrast
    - changing hue and saturation levels
    - Color Jitter
    -Random Rotation
    
![aug.png](/images/aug.PNG) 


2. Chose size as 224 * 224 for input tensor by resizing and cropping it using transform

3. Normalised the input images
```
transforms.Resize((224,224))
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])


```

## Models Used :

   ### 1. CNN with Transfer Learning : Used ResNet50 Model
   - Frozen Layers : conv1 , bn1 , relu, maxpool, layer1, layer2, layer3, avgpool,layer4
   - Unfrozen Layers :  fc
        
        Residual networks is based on theory that as we go deeper, we should make sure not to degrade accuracy and so, Keep learning the residuals to match the predicted with the actual and this is acheived by mapping the identity function.

        **This architechture contains**
        - (conv1) as first convolutional layer containing in channels as 3 which is due to RGB input tensor
        - (bn1) as batch normalization layer
        - followed by ReLU and MaxPooling
        - then it contains 4 main layers named layer1, layer2, layer3 and layer4 which contains further sub layers of convolution
        - followed by batchnorm
        - followed by relu
        - followed by maxpooling
        - and then finally fc.
        
        ReLU activation is used as it's the most proven activation function for classification problems as it introduces good and right amount of non linearity with less chances of vanishing gradient problem ! Batch normalization helped in making the network more stable and   learning faster thereby faster convergence. Maxpooling helped in downsampling high number of parameters created by producing higher dimensional feature maps after convolution operation and thus selecting only relevant features from the high dimensioned feature matrix.

   **Last Layer is Modified as follows :**
          ```
          Linear(in_features=2048, out_features=512) 
          nn.Linear(516,64)
          nn.Linear(64,3)
          ```
       
with ReLU activations between the linears.
    
 ``` 
  Optimiser : SGD
 Loss Function : CrossEntropyLoss()
 ```
## Results


```
Test Loss: 0.518940

Test Accuracy: 100% ( 5/ 5)
```
![res1.png](/images/res1.PNG) 
![res2.png](/images/res2.PNG) 
![res3.png](/images/res3.PNG) 

## Deployment
I used Flask to deploy my trained model 
 - importing the model with checkpoints and transforming image to get tensor done in **_commons.py file._**
 - Passing the test image through trained model is done in **_inference.py_**

##  Occlusion Heat Map
Visualizing what CNN is learning using Occlusion technique. 
![heat.png](/images/heat.PNG) 

Investigating which part of the image some classification prediction is coming from is by plotting the probability of the class of interest as a function of the position of an occluder object. That is, we iterate over regions of the image, set a patch of the image to be all zero, and look at the probability of the class.





        
            

    
   
    

  
  
  
