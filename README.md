# PhoneNumber Reader Using AI (OCR Model)
A Python-based optical character recognition (OCR) system was developed specifically for extracting phone numbers. This OCR model was trained exclusively on synthetic images, devoid of real-world data, yet it demonstrates efficacy when applied to real-world scenarios.


## Steps to run this project

#### First: install required libraries
You need to install these important libraries to run the project:

    $ pip install tensorflow
    $ pip install opencv-python

#### Second: Generate the training data by running the generator file
This file generates phone numbers written on an image with distributed variation to prevent model from overfitting

    $ python3 generator.py

#### Third: Start training the model
By running the the train file, the models starts to train it self with generated data

    $ python3 train.py

#### Fourth: Start testing the model
By running the test file you are able to test your model on any kind of image that contains just a phone number 

    $ python3 test.py