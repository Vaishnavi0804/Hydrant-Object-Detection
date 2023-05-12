# Computer Vision: Hydrant Object Detection
DAEN-690-002 : Team Phoenix working with the Fairfax County Fire and Rescue Department

Hydrant Object Detection is a deep learning project that aims to detect and classify fire hydrants in images using Convolutional Neural Networks (CNNs). The project is built using Python and TensorFlow, a popular open-source library for machine learning.

## Dataset
The project includes a dataset of street view images from the year 2018 to 2020 provided by the Virginia Department of Transportation (VDOT). CNN model was tested and trained on labelled data, where all the images with hydrants were manually annotated using CVAT annotation tool. The dataset includes images of fire hydrants from different angles, distances, and lighting conditions. The annotations include the bounding boxes and classes of the hydrants in each image.

## Image Preprocessing
Since this is an image dataset, data for EDA was parsed from the XML files after annotations into a dataframe, this code is available in the 'data_parsing.ipynb' file. Additionally, images are cropped to a specific format to be ready for model training, this code is available in the 'Preprocessing_step-1.ipynb'.

## Model
The code includes a script for training the model, which uses the pre-trained CNN model. The model architecture includes several convolutional and pooling layers, followed by several fully connected layers, and a softmax output layer.

## Usage
To use the pre-trained model to detect fire hydrants in new images, run the 'Model Training,testing & Evaluation 2.ipynb' script. The script takes an image file as input and outputs an image with the detected hydrants in boundary boxes along with their class and confidence level.

## Installations
Entire project was developed in Hopper Cluster. So all the resources needed like TensorFlow, CUDA versions and CUDNN were installed in the cluster. 

## Training
Both EfficientDet and Faster R-CNN models are trained on our dataset of manually annotated images with and without hydrants.

## Evaluation
Created a confusion matrix which includes precision and recall of both the models, on comparing both the models, Faster R-CNN was identified as the suitable model for object detection because it had good precision and recall score, 95% and 76% respectively.

## Conclusion
Hydrant Object Detection is a useful tool for firefighters, city planners, water quality inspector, etc. who need to locate and map fire hydrants in their communities to respond to fire emergencies. The project report is well-documented and available here, making it accessible to anyone with an interest in deep learning and object detection.
