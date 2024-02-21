Bachelor thesis
-----
A system for automatic face recognition from photographs based on machine learning
-----
In this thesis, a system for automatic facial recognition from photos based on machine learning is made. 
The system allows users to do the training of the system itself through the console by providing it with
a folder that contains selected images. It also allows testing on a completely new image and as a result
gives the identity of the person shown by the probabilities in percentages. The principles of operation
of neural networks that are the basis of the functioning of this system are also explained.
Visual studio code environment, Python programming language, FaceNet system and various libraries described
in the paper were used in the development of the system. The motivation for the creation of this system is
huge precisely because of the possibility of application for various purposes.
-----
1. Firstly install requirements from requirements.txt
2. train with python3 -m training.train -d something/faceRecognition/images
3. test with python3 -m inference.classifier_check --image-path something/faceRecognition/test_images/108.jpg --save-dir something/faceRecognition/results
4. check results in results folder