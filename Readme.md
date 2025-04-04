# Face Emotion Recognition using Machine Learning in Python

This project implements a facial emotion recognition system using machine learning techniques in Python. The system detects human emotions from facial expressions in real-time video streams or static images.

## Overview

Facial emotion recognition is a critical area in computer vision, aiming to interpret human emotions from facial expressions. This project utilizes a Convolutional Neural Network (CNN) to classify images into seven distinct emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral. The system can process both real-time video feeds and static images.

## Dataset

The model is trained on the FER-2013 dataset, which contains 35,685 grayscale images of 48x48 pixels, each labeled with one of the seven emotions. The dataset is divided into training and testing subsets.

## Model Architecture

The CNN model comprises the following layers:

1. **Convolutional Layers**: Extract features from input images.
2. **MaxPooling Layers**: Downsample feature maps to reduce dimensionality.
3. **Dropout Layers**: Prevent overfitting by randomly setting a fraction of input units to zero during training.
4. **Flatten Layer**: Convert 2D matrices to a 1D vector.
5. **Fully Connected (Dense) Layers**: Perform high-level reasoning based on the extracted features.
6. **Output Layer**: Classifies the input into one of the seven emotion categories using a softmax activation function.

## Dependencies

Ensure the following libraries are installed:

- Python 3.x
- NumPy
- Pandas
- Keras
- TensorFlow
- OpenCV
- scikit-learn
- tqdm
- Jupyter Notebook

## Installation

1. **Clone the Repository**:

   git clone https://github.com/kumarvivek9088/Face_Emotion_Recognition_Machine_Learning.git

2. **Install Required Libraries**:

   pip install numpy pandas keras tensorflow opencv-contrib-python scikit-learn tqdm jupyter

## Usage

1. **Navigate to the Project Directory**:

   cd Face_Emotion_Recognition_Machine_Learning

2. **Launch Jupyter Notebook**:

   jupyter notebook

3. **Open and Run the Notebook**:

   - Open `Face_Emotion_Recognition.ipynb`.
   - Execute the cells sequentially to train the model and test it on sample images or real-time video feeds.

## Results

The model achieves an accuracy of approximately 62% on the test dataset. Performance can be further improved by:

- Fine-tuning the model architecture.
- Experimenting with different hyperparameters.
- Utilizing data augmentation techniques to increase dataset diversity.