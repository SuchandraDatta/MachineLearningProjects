**Creating a classifier for dogs and cats using a CNN**

I wrote a detailed article on this topic at towardsdatascience linked as follows: https://towardsdatascience.com/beginners-guide-to-transfer-learning-on-google-colab-92bb97122801

1. Dataset used is the one provided in the Udemy course Machine Learning A-Z which has been converted to a csv file with the following format: Each row represents 1 image of size 64X64 converted to grayscale. Last number of each row is the class label encoding 1=cat, 0=dog.
2. Dataset has total 10000 images(5000 dogs, 5000 cats)
3. Hyperparameter tuning: Manual. Screenshot of final training shows training accuracy of 93% and testing accuracy of 83%

**Application of transfer learning**
Transfer learning can be applied in the following ways
1. Use the pre-trained weights as it is without any training or tuning
2. Use as a feature extractor: Remove the final layer of the model, add custom ANN or CNN and train the model, freezing the previous layers so that already learnt features aren't updated during training.
3. Fine tuning: Unfreeze all layers and train

Observations:
1. Fine tuning works much better than using as it is or using as a feature extractor.
2. VGG16 works best, 91% testing accuracy. Able to classify lions as cats and wolves as dogs. Classifies cartoon images also correctly. 
3. ResNet50, MobileNet and any network that has the batch normalization layer is tough as the dataset's mean and standard deviation don't match that of the original dataset on which it had been trained, so those layers have to be unfrozen and if all layers are unfrozen then it works better. 
4. Batch sizes of <=128 aren't much good, the model accuracy during training increases and then decreases and then increases again. 
5. Models expect RGB images and have a minimum acceptable image size.

Results from VGG-16
![VGG16](https://user-images.githubusercontent.com/41965125/104593110-35f07900-5695-11eb-9913-d0ad6d934c44.png)



Results from ResNet-50
![ResNet50](https://user-images.githubusercontent.com/41965125/104593162-486ab280-5695-11eb-9093-01d4fe498895.png)


Results from model made from scratch
![Model_from_Scratch](https://user-images.githubusercontent.com/41965125/104593232-61736380-5695-11eb-9bfd-efa50546fe4b.png)