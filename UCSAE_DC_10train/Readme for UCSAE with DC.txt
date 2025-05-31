UCSAE is an unsupervised feature learning framework for hyperspectral remote sensing imagery classification. 
For the unsupervised sparse autoencoder feature learning, the ground truth and the whole image can be used for unsupervised SAE feature learning without any label.

There are two choices you can train the UCSAE:
1. Ground truth patch training, run SparseAE_SingleLayer.m
2. Whole image patch training, run SparseAE_SingleLayer_scale2.m

The main code is testing upon the DC dataset, and the code the data for PU is attached in a link.

Data for this paper:
Data for this work can be downloded from the given link，just unzip the DC10_train.

SVM evaluation:
For the evaluation by SVM, run Classify_RBF_UCSAE_DC.m or run the file in the given svm folder.
