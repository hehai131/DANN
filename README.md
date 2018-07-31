## DANN
This is a implementation of [Domain-Adversarial Training of Neural Networks][1] with pytorch.

## Dataset
In this work, Mnist, Mnist-m, svhn digitals dataset and office-31 dataset is used to train our network. You can get the dataset by yourself.

## experiment

|Method     | Mnist -> Mnist_m | Mnist -> svhn| Svhn -> Mnist|
|:----------:|:-----------------:|:---------------------:|:---------------------:|
|Source Only| 27.053632            | 18.927479| 81.633333|
|DANN       | 92.512208            | 24.750164| 67.900000|``````

[1]:https://arxiv.org/pdf/1505.07818.pdf
