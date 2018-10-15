# Zero-shot-learning-journal

This repository provides codes and data of our papers:

[1] Soravit Changpinyo, Wei-Lun Chao, and Fei Sha, ["Predicting visual exemplars of unseen classes for zero-shot learning,"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Verma_Generalized_Zero-Shot_Learning_CVPR_2018_paper.pdf) ICCV, 2017

[2] Wei-Lun Chao*, Soravit Changpinyo*, Boqing Gong, and Fei Sha, ["An empirical study and analysis of generalized zero-shot learning for object recognition in the wild,"](https://arxiv.org/pdf/1605.04253.pdf) ECCV, 2016

[3] Soravit Changpinyo*, Wei-Lun Chao*, Boqing Gong, and Fei Sha, ["Synthesized classifiers for zero-shot learning,"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Changpinyo_Synthesized_Classifiers_for_CVPR_2016_paper.pdf) CVPR, 2016 

Note that the codes for [3] are largely based on another repository [zero-shot-learning](https://github.com/pujols/zero-shot-learning).

# Installation
1. Download the following packages:
* [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
* [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
* [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
* [multicore-liblinear](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear/)

2. Unzip and put them in the folder /tool, and compile. For libsvm, liblinear, multicore-liblinear, you only need to compile the /matlab subfolder.

3. Check the paths and *change* the folder names.
* Now in /tool, you should have 4 folders: /minFunc, /libsvm, /liblinear, /liblr-multicore. 
* In /minFunc, you should immediately see 3 subfolders and 4 .m files.
* In /libsvm, /liblinear, /liblr-multicore, you should immediately see the /matlab subfolder.

# Data
1. For AwA, CUB, and SUN:
* Download the [googleNet features](https://www.dropbox.com/s/7h1dta59ocdptlu/googleNet_features.zip?dl=0). Unzip and put the three .mat files in the data folder.
* Download the resnet features and class splits from [Yongqin Xian's website: NS (PS)](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), [SS](http://datasets.d2.mpi-inf.mpg.de/xian/standard_split.zip). Unzip and put the xlsa17 and standard_split folders in the data folder. Run data_transfer.m to generate 8 .mat files ended with "_resnet.mat".

# Running the codes


# Evaluation
