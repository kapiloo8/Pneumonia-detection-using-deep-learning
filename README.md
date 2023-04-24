# Pneumonia-detection-using-deep-learning
In this article, we will discuss solving a medical problem i.e. Pneumonia which is a dangerous disease that may occur in one or both lungs usually caused by viruses,
fungi or bacteria. We will detect this lung disease based on the x-rays we have.
Chest X-rays dataset is taken from Kaggle which contain various x-rays images differentiated by two categories “Pneumonia” and “Normal”.
We will be creating a deep learning model which will actually tell us whether the person is having pneumonia disease or not having pneumonia.
Tools:
VGG16: It is an easy and broadly used Convolutional Neural Network (CNN) Architecture used for ImageNet which is a huge visible database mission utilized in visual object recognition software research.

Transfer learning (TL): It is a technique in deep learning that focuses on taking a pre-trained neural network 
and storing knowledge gained while solving one problem and applying it to new different datasets.
In this article, knowledge gained while learning to recognize 1000 different classes in ImageNet could apply when trying to recognize the disease.
Modules Required:

Keras: It is a Python module for deep learning that runs on the top of TensorFlow library. 
It was created to make implementing deep learning models as easy and fast as possible for research and development. Being the fact that Keras runs on top of Keras we have to install TensorFlow first. 
To install this library, type the following commands in IDE/terminal.

SciPy: SciPy is a free and open-source Python module used for technical and scientific computing. 
As we require Image Transformations in this article we have to install SciPy module. To install this library, type the following command in IDE/terminal.

glob: In Python, the glob module is used to retrieve files/pathnames matching a specified pattern. 
To find how many classes are present in our train dataset folder we use this module in this article.
