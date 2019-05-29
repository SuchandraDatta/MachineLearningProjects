**REVIEW CLASSIFICATION USING TEXT MINING**
1. This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015. It consists of 
reviews of products, movies, restaurants with each row being a sentence separated from it's label(1 for positive, 0 for negative) by a tab.

2. Basic preprocessing done include removal of all punctuation, convert to lowercase, split into words, remove stop words, perform stemming,
rejoin words with a space to create corpus.

3. Tokenizer used to vectorize the corpus. Length of sentences are different hence size of each term in x_train or x_test will be different
creating problems in input to ANN or CNN so padding is applied.

4. CNN created with Embedding layer.


![pic1](https://user-images.githubusercontent.com/41965125/58557785-dbdf3d80-823c-11e9-9f9e-27efd8bf06f6.png)

Screenshot of final training and testing accuracy achieved
