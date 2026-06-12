# Deep Learning Coursework

This repository contains deep learning coursework experiments for CSC3066 Deep Learning. The projects explore neural network implementation, model evaluation, and hyperparameter tuning across tabular and text classification tasks.

## Projects
## 1. German Credit Classification

A fully connected multi-layer perceptron was implemented from scratch using NumPy and trained on the German Credit dataset. The model uses mini-batch gradient descent, backpropagation, sigmoid activations, and binary cross-entropy loss to classify credit risk.

The experiment also investigates the effect of different learning rates and epoch counts on model performance.

## 2. TREC Question Classification

Several neural architectures were evaluated for multi-class question classification using the TREC dataset. The models include:

MLP with TF-IDF features
MLP with pre-trained GloVe embeddings
CNN with word embeddings
RNN/LSTM with word embeddings

Model performance was compared using test accuracy, categorical cross-entropy loss, training curves, and confusion matrices. The CNN achieved the best overall result, reaching 90.20% test accuracy.

## Key Topics
Neural networks from scratch
Backpropagation
Mini-batch gradient descent
Binary and categorical classification
TF-IDF and word embeddings
CNN and RNN architectures
Hyperparameter tuning
Model evaluation and overfitting analysis
Technologies Used
Python
NumPy
TensorFlow / Keras
Scikit-learn
Matplotlib
GloVe embeddings