#  Product Identification using Deep Learning  

**Input To WebApp** : Upload Image from Local Machine and Submit
![pg1.png](/images/pg1.png) 

**Result On WebApp** : Breed Name is Displayed for Dogs and Closest breed of dog for Human inputs
![pg2.png](/images/pg2.png) 


## Dataset Used :    
[Data Set Provided](/images) 



## Preprocessing 

I choose size as 224 * 224 for input tensor by resizing and cropping it using transform
```
transforms.Resize(256)
transforms.CenterCrop(224)
```
Yes, data augmentations can help in increase the training examples on the fly during training and help to increase the performance of architechture due to randomly indtroduced variations of the sampled examples which i have implemented here using transforms
```
transforms.RandomRotation(30)
transforms.RandomResizedCrop(224)
transforms.RandomHorizontalFlip()
```



## Models Used :
  ### 1.CNN From Scratch :
  ```
  Net(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=50176, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=133, bias=True)
  (dropout): Dropout(p=0.5)
)
  ```

  
   
   ### 2. CNN with Transfer Learning : Used ResNet50 Model
   - Frozen Layers : conv1 , bn1 , relu, maxpool, layer1, layer2, layer3, avgpool
   - Unfrozen Layers : layer4 and fc
        
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
        Linear(in_features=2048, out_features=512) Linear(in_features=512, out_features=133) with ReLU activations between the linears.
        I choose to first only train *layer4* and *fc* and then i unfroze one more layer *layer3* which increased the accuracy and decreased the validation losses.
 ``` 
  Optimiser : SGD
 Loss Function : CrossEntropyLoss()
 ```
## Results
CNN from scratch acheived very low accuracy but when we used a pretrained model like ResNet wih tranfer learning , results were very good.

```
Test Loss: 0.343228

Test Accuracy: 89% (747/836)
```
![re1.png](/images/re1.png) 
![re2.png](/images/re2.png) 

## Deployment
I used Flask to deploy my trained model 
 - importing the model with checkpoints and transforming image to get tensor done in **_commons.py file._**
 - Passing the test image through trained model is done in **_inference.py_**


## References
- [How to Deploy model using Flask](https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717)
- [Loss Functions and Optimizers](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)
- [How ResNet works](https://medium.com/@pierre_guillou/understand-how-works-resnet-without-talking-about-residual-64698f157e0c)



        
            

    
   
    

  
  
  
