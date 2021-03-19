# Deep Learning

Sheil Sarda

`<sheils@seas.upenn.edu>`

Taught by Prof. Pratik Chaudhari in Fall 2020

## [Brief Summary of Topics Covered](refs/course_summary.pdf) 

## Assignments Overview

**Note:** PS0 was just a refresher for the Calculus and Statistics concepts required for the class

### PS1 SVM and Neural Networks

- Fit MNIST dataset using a SVM from `scipy`
- Prove Jensen's inequality
- Train a neural network completely from scratch on the MNIST dataset

### PS2 ReLU, CNNs and LSTMs

- Examine the ReLU architecture in `pytorch`
- Understand non-convex optimization
- Understand how batch normalization relates to gradient descent updates
- Train a convolutional neural network on the CIFAR-10 dataset
- Understand the vanishing gradient problem and how the LSTM architecture solves this

### PS3 Cauchy-Schwartz inequality and RNNs

- Use the Cauchy-Schwartz inequality to prove that the co-coercivity of the gradient and its Lipschitz continuity are equivalent 
- Implement a RNN for predicting the next character in a given sentence using Leo Tolstoy's War and Peace as the training dataset

### PS4 Logistic Regression and CNN Hyperparameter tuning

- Implement logistic regression for classifying zero and one from MNIST only using `numpy`
- Train the All-CNN network on CIFAR-10 with a custom learning rate schedule

### PS5 Variational Inference and VAEs

- Understand the theory behind Variational Inference
- Train a Variational Auto-Encoder for generating MNIST digits

## [Final Project: Few-Shot Meta Q-Learning with Reptile](https://github.com/sheilsarda/ESE546_Final_Project)