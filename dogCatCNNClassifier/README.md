**Creating a classifier for dogs and cats using a CNN**
1. Dataset used is the one provided in the Udemy course Machine Learning A-Z which has been converted to a csv file with the following format: Each row represents 1 image of size 64X64 converted to grayscale. Last number of each row is the class label encoding 1=cat, 0=dog.
2. Dataset has total 10000 images(5000 dogs, 5000 cats)
3. Hyperparameter tuning: Manual. Screenshot of final training shows training accuracy of 93% and testing accuracy of 83%

**Transfer Learning**<br/><br/>
Tried out transfer learning on different models. Inception and Xception could not be used as it expects a minimum image of 71X71 and 75X75 whereas image size at hand is only 64X64. All pretrained models also expect an RGB image as input. Models with batch normalization layer like ResNet50, MobileNet didn't do well on freezing layers as the frozen layers contain mean values of the original dataset it was trained on, not the current one fed into it. They also didn't get very high testing accuracy on frozen layers(it was stuck at around 80%)but fine-tuning yeilded best results. Increasing batch size fixed the problem of increasing and then decreasing accuracyvalues.<br/>
So transfer learning was applied<br/>
:herb:Directly using the pre-trained weights<br/>
:herb:Freezing upper layers, adding custom ANN and training<br/>
:herb:Unfreezing all layers and fine-tuning
<br/>
Best model performance was got with VGG16.


