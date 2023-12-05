# ML2-Project

## Authorship

- **Authors:** Antónia Brito, António Cardoso, Pedro Sousa
- **University:** Faculty of Science from University of Porto
- **Course:** Machine Learning II (CC3043)
- **Date:** 05/12/2023

## Description

This project is focused on the development of deep learning models for audio classification.  
The data used to design and build the models is found in the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) , which was thouroughly used during the development of this project. This dataset contains a total of 8732 labeled audio recordings of urban sounds, each with a duration of up to four seconds. Each excerpt has been labeled with one of the following classes:
- air conditioner
- car horn
- children playing
- dog bark
- drilling
- engine idling
- gun shot
- jackhammer
- siren
- street music

The objective of this project relied on defining, compiling, training and evaluating two Deep Learning (DL) classifiers. The DL model types to be considered were:
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)

Furthermore, it was asked to realize performance evaluation on both constructed models by running 10-fold cross validation with the 10 predefined folds that come in the dataset.

Finally, an experiment was conducted with the goal of evaluating each model's robustness against adversarial examples by implementing the algorithm _[DeepFool](https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf)_.

## Solutions Implemented

- Convolutional Neural Network
- Recurrent Neural Network

## Project Development Phases

### Convolutional Neural Network
- **Data Pre-Processing**
- **Feature Extraction**
  + Mel-scaled Spectrograms (2D arrays)
  + Chromagrams (2D arrays)
  + Spectral Flatness, Bandwidth, Roll-off, Centroid (1D arrays stacked)
- **CNN Architecture Definition**
- **Performance Assessment**
- **Architecture Changes for Overfit Prevention**

### Recurrent Neural Network
- **Data Pre-Processing**
- **Feature Extraction**
  + Log Mel-Scaled Spectrograms (2D arrays)
- **CNN Architecture Definition**
- **Performance Assessment**
- **Architecture Changes for Overfit Prevention**

### Performance Evaluation
- 10-fold Cross Validation
- Mean and Standard Deviation values of the accuracies obtained in each iteration
- Confusion Matrix

### Robustness Evaluation Against Adversarial Examples
- Implementation of the _DeepFool_ algorithm
- For each model of the iterations of the Cross Validation run, obtain the model's robustness using the results computed with _DeepFool_ for each example in the corresponding test fold

## Python and Libraries Versions

| Python | 3.10.13 |
|:------:|:-------:| 

| Library | Version |
|:--------|:--------|
| ipykernel | 6.25.2 |
| keras | 2.10.0 |
| librosa | 0.10.1 |
| matplotlib | 3.7.2 |
| numpy | 1.26.0 |
| pandas | 2.1.3 |
| soundfile | 0.12.1 |
| tensorflow | 2.10.0 |
