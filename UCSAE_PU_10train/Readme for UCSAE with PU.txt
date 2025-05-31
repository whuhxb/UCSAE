UCSAE is an unsupervised feature learning framework for hyperspectral remote sensing imagery classification. 
For the unsupervised sparse autoencoder feature learning, the ground truth and the whole image can be used for unsupervised SAE feature learning without any label.

There are two choices you can train the UCSAE, but for the PU dataset, due to the image size and memory consumption, mainly utilizing the ground truth patch training,
run SparseAE_SingleLayer_scale2.m, can also run SparseAE_SingleLayer.m. If interested in whole image patch training, can generate the whole image patch and run a large 
CPU memory size machine.

Data for this paper:
Data for this work can be downloded from the given link，just unzip the PU10_train.zip.

SVM evaluation:
For the evaluation by SVM, run Classify_RBF_UCSAE_PU.m or run the file in the given svm folder.