
# Transfer Learning in Image Classification

Welcome to my Flower Recognition Project Repository!
In this project, I've developed a custom Convolutional Neural Network (CNN) named `ResNet9` for flower recognition. The CNN was initially trained on the `"Flower Recognition"` dataset by ALEXANDER MAMAEV, which is avaible on Kaggle and comprises 4317 images of flowers categorized into five classes: sunflower, tulip, rose, dandelion, and daisy. Through this initial training, the model achieved an accuracy of 82%.
Subsequently, I implemented transfer learning by loading a pre-trained ResNet34 model and fine-tuning it on the same flower dataset. This resulted in a significant accuracy boost, reaching an impressive 94.7%. The dataset was split into a validation set with 431 images and a training set with 3886 images.



## Network architecture 

### ResNet9


The ResNet9 consists of convolutional blocks (conv_blocks) with batch normalization and ReLU activation functions. Two residual blocks (res1 and res2) are incorporated to enhance feature learning, and max-pooling layers are used for downsampling. The classifier section includes an adaptive max pooling layer followed by flattening and a linear layer for the final classification. The forward method outlines the flow of data through the network. This ResNet9 architecture is designed for image classification tasks, with flexibility and improved performance achieved through the integration of residual blocks.

**ResNet9**
![ResNet9](https://i.ibb.co/RTRZZ4j/image.png)


### ResNet34

ResNet34 is a variant of the ResNet (Residual Network) architecture introduced by Microsoft Research in 2015. The "34" in ResNet34 represents the total number of layers in the network. This architecture is known for its deep structure, utilizing skip connections or residual blocks to overcome the vanishing gradient problem during training of very deep networks.

ResNet34 has demonstrated remarkable performance by achieving high accuracy on various datasets. It is a pre-trained model that has been trained on a large dataset (e.g., ImageNet) and can be fine-tuned for specific tasks with transfer learning.


**ResNet34**

![test](https://i.ibb.co/WtG7hJH/The-structure-of-the-Res-Net34-CNN-Network-The-input-of-the-network-is-the-preprocessed.png)

*I have aligned  my image dataset to ImageNet dataset since ResNet34 is trained on ImageNet*

#### Transfer learning
Transfer learning is a machine learning technique where a model trained on one task is repurposed or fine-tuned for a different, but related, task. Instead of training a model from scratch, transfer learning leverages knowledge gained from solving one problem to improve performance on a different, but related, problem.

I employed transfer learning to fine-tune the pre-trained ResNet34 model. Thus i used the ResNet34 architecture as a feature extractor. Finally i replaced the final fully connected layer of the original ResNet34 model with a new linear layer to adapt the model for the target classification problem with a specified number of output classes.(i.e. 5)


## Training and Evaluation

#### ResNet9
The training and evaluation process is accomplished using the provided utility functions. The model is trained in two phases, each comprising 10 epochs (you can experiment there is not hardandfast rule for hyperparameters). In the first phase, a learning rate of 0.001 is employed reaching an accuracy of `67%`, followed by a second phase with a reduced learning rate of 0.0001 improved the accuracyto `82%`. The Adam optimizer is utilized for optimization during both phases. 



**Metrics; first 10 epochs with lr=0.001**

![Metrics](https://i.ibb.co/fM6ZP2g/image.png)

**Metrics; second 10 epochs with lr=0.0001**

![Metrics](https://i.ibb.co/g4yT4F9/image.png)

**Accuracy plot**

![Accuracy](https://i.ibb.co/nB0d8sq/image.png)

**Loss plot**

![Loss](https://i.ibb.co/jkLcZTB/image.png)



#### ResNet34
The training process incorporates a one-cycle learning rate policy, which dynamically adjustes the learning rate during training. The training loop involves iterating through batches of the training dataset, computing the loss, and updating the model parameters through backpropagation. Additionally, gradient clipping is applied to prevent exploding gradients.

The training process is monitored using a validation set, and model performance metrics, including loss and accuracy, were evaluated. The learning rate is adjusted using the OneCycleLR scheduler, contributing to effective training. 

The training was done for 5 epochs, at the end of 5th epoch accuracy reached `94.7%`

**Metrics**

![Metrics](https://i.ibb.co/J3t4mX9/image.png)


**Accuracy plot**

![Accuracy](https://i.ibb.co/SPryLHG/image.png)

**Loss plot**

![Loss](https://i.ibb.co/X44VCd6/image.png)

**Learning rate finder**

![LR](https://i.ibb.co/5xmW9bF/image.png)
# Results 

### Testing on test set (Images from internet)

**Dandelion predicted correct**
![Dandelion](https://i.ibb.co/PwTYrvj/image.png)


**sunflower predicted correct**
![sunflower](https://i.ibb.co/z8NrJFB/image.png)

**Rose predicted correct**
![Rose](https://i.ibb.co/fFwSmds/image.png)

**Daisy predicted correct**
![Daisy](https://i.ibb.co/XVSGSvm/image.png)

**Tulip predicted correct**
![Tulip](https://i.ibb.co/N6gqP1L/image.png)


### Comparision 

**ResNet9 prediction**
![incorrect](https://i.ibb.co/VvcYTXd/image.png)

**ResNet34**
![correct](https://i.ibb.co/Nx7Dg8X/image.png)

