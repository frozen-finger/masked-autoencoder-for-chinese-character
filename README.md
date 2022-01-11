# masked-autoencoder-for-chinese-character
Masked Autoencoder (Chinese font reconstruction and font classification 

This Masked Autoencoder is created according to the paper Masked Autoencoders Are Scalable Vision Learners.
And we added another task in pre-training.

You must run imagegeneration.py firstly in Data to get font image.
Front image is created according to the Unicode. To make sure you create the right image, you may need to download required font file.
Then you can run clear.py to get rid of the useless image(which is all white).


Change the hyper-parameters in train.py, and run train.py.
