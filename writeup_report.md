#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the Nvidia architecture with Batch norm layers and drop out added to prevent overfitting. Batch norm can also help generalize.

#### 3. Creation of the Training Set & Training Process

The provided training data is not enough. I captured an additional lap for each direction.

Car still drove to the curb sometimes. I captured additional data to restore the control when car approches too close to the curbs.

I used data augmentation to flip all the images.

I finially random shuffled the data set and put 20% to the validation set.

and adam optimizer was used to train the model. The default learning rate works, so I didn't tune this hyper-parameter.

This model works for this track and is probably over-fitted to this track.To generalize on other tracks, more training data is needed.


