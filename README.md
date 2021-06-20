# Handwritten Digit Recogniser using PyTorch and OpenCV

## Introduction

▪ The goal of this project is to recognize the handwritten digit shown in front of the camera and to display it.

▪ A convolutional neural network is trained to transcribe digits. The data from the learning stage allows the Pi Camera to read and recognize the digits.

▪ To implement this, Scikit and OpenCV are used for image processing and PyTorch is used for deep learning.

## Algorithm

STEP 1: Open the terminal in RPi and run the ‘Digit_Recognizer.py’ file. This will automatically start an input video frame.   
          
STEP 2: Show a number in the RPi camera. This will be scaled later since the convolutional neural network expects images of a certain size.

STEP 3: Convert the acquired image to gray-scale by using the scikit-image function call.

STEP 4: Then convert the image from a floating point format to a uint8 range [0, 255]. To obtain a black and white image, thresholding is done via the Otsu method. 

STEP 5: Resize the image to a 28x28 pixel array. This is then flattened to a linear array of size (28x28). 

STEP 6: Invert the image since the model accepts images as 28x28 pixels, drawn as white, on black background.

STEP 7: The deep neural network weights and feeds the image to the network.

STEP 8: Display the predicted answer on the output video frame.

STEP 9: For the prediction of the next digit proceed from step 1 to 8.

STEP 10: Activate the stop command to terminate the program.

![Screenshot](/docs/Screenshot.png)

## Result

The overall validation accuracy of 97% is obtained in the recognition process by the Convolution Neural Network.
