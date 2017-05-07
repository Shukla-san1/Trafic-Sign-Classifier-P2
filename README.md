# Trafic-Sign-Classifier-P2
This project is build for  Traffic Sign Recognition using CNN.
# Installation:
* Refer [Link](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) to se detailed steps for setting up enviroments for execution.
* This will setup your computer to run the code on your computer.
* Now you clone the project if not done already.
* whole code is written in ipython notebook, so one can open it by running *$ jupyter notebook Traffic_Sign_Classifier.ipynb*

# Usage:
* $ jupyter notebook Traffic_Sign_Classifier.ipynb
* Now go to individual cell and ipython notbook and run in different cell.


# Contributions:
As we had used CNN to classify the trafic sign, so having good set of training data is required. Image Augmentation technique used in this prject
can be further enhanced. As of now, I have done following augmentation.
* Blurring of image, used opencv python api(cv2.GaussianBlur) for the same.
* Rotation of iamges by 2 degree,cv2.getRotationMatrix2D
* Tranlation of images.
There is scope here to improve, So the trained model should be able to classify the trafic sign in different whether condition, different light
condition, even if there is shadow. 
