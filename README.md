# Hydrant-Object-Detection
DAEN-690-002 : Team Phoenix working with the Fairfax County Fire and Rescue Department

Hydrant Object Detection is a deep learning project that aims to detect and classify fire hydrants in images using convolutional neural networks (CNNs). The project is built using Python and TensorFlow, a popular open-source library for machine learning.

## Dataset
The project includes a dataset of annotated images of fire hydrants, which was used to train and test the CNN model. The dataset includes images of fire hydrants from different angles, distances, and lighting conditions. The annotations include the bounding boxes and classes of the hydrants in each image.

## Model
The code includes a script for training the model, which uses transfer learning to leverage the pre-trained weights of the VGG16 CNN model. The model architecture includes several convolutional and pooling layers, followed by several fully connected layers, and a softmax output layer. The model was trained using the Adam optimizer and the cross-entropy loss function.

## Usage
To use the pre-trained model to detect and classify fire hydrants in new images, run the detect_hydrants.py script. The script takes an image file as input and outputs an annotated image with the detected hydrants and their classes.

## Installation
To install the required dependencies, run the following command:

## Training
To train the model on the dataset, run the train_model.py script. The script takes the path to the dataset as input, and saves the trained model to a file.

## Evaluation
To evaluate the performance of the trained model on the test set, run the evaluate_model.py script. The script calculates the accuracy, precision, recall, and F1-score of the model.

## Conclusion
Hydrant Object Detection is a useful tool for firefighters, city planners, and others who need to locate and map fire hydrants in their communities. The project is well-documented and easy to use, making it accessible to anyone with an interest in deep learning and object detection.
