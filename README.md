# Deep Learning (PyTorch)

This repository contains material related to Udacity's [Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by real people (Udacity reviewers), but the starting code is available here, as well.

# UDACITY’S INTELLECTUAL PROPERTY RIGHTS
The Services are owned and operated by Udacity and its licensors. All content or other material available through the Services, including, but not limited to, information on the Websites, on-line lectures, speeches, video or other lessons, applications, pre-assessments and assessments, quizzes, presentation materials, homework assignments, programming assignments, programs, code, licenses, and other images, text, layouts, arrangements, displays, illustrations, documents, surveys, materials, audio and video clips, HTML, and files (collectively, the "Content"), are the property of Udacity and/or its affiliates or licensors and are protected by copyright, patent and/or other proprietary intellectual property rights under United States and foreign law.

Udacity logos, trademarks and service marks which may appear on the Website(s), throughout the Services and in Online Courses ("Marks"), are the property of Udacity and are protected under United States and foreign laws. All other trademarks, service marks, and logos used on the Services, with or without attribution, are the trademarks, service marks, or logos of their respective owners. In addition, elements of the Services are protected by trade dress and other federal and state intellectual property laws and may not be copied, reproduced, downloaded or distributed in any way in whole or in part without the express written consent of Udacity.

From time to time, Udacity may include software, code, instructions, or other such information in the Content or materials for the Services; any such information is provided on an "as-is" basis for instructional purposes only and is subject to the Disclaimer of Warranties and Limitation of Liabilities sections below and other terms herein. Any use of such information for commercial purposes is strictly prohibited. Udacity and/or its affiliates and licensors reserve all rights not expressly granted herein to the Services, Content, and Marks.

# LICENSE TO USE THE SERVICES
Subject to your compliance with this Terms of Use, Udacity hereby grants you a freely revocable, worldwide, non-exclusive, non- transferable, non-sublicensable limited right and license (a) to access, internally use and display the Services, including the Content, at your location solely as necessary to participate in the Online Courses as permitted hereunder, and (b) to download the Educational Content (as defined below) so that you may exercise the rights granted to you under the section entitled "Creative Commons license" below. You must abide by all copyright notices or restrictions contained on the Websites, in any Online Course, or in the Content. You may not delete any attributions, legal or proprietary notices on the Websites, in the Online Courses, or on the Content.

In some instances, some of the Content belonging to our licenses and provided in connection with the Online Courses, may require you to agree to that licensor’s terms and conditions. Udacity has no control and is not responsible for those third-party services as further described in the Linking to Third Party Sites and Content section below.

# LICENSE TO EDUCATIONAL CONTENT
As used herein, "Educational Content" means a subset of the Content containing the educational materials made available to you through the Online Courses, including such online lectures, speeches, video lessons, quizzes, presentation materials, homework assignments, programming assignments, code samples, and other educational materials and tools. Such Educational Content will be considered the "Licensed Material" under the terms of the CC License (as defined below).

Udacity hereby grants you a license in and to the Educational Content subject to, as modified herein, the terms and conditions of the Creative Commons Attribution-NonCommercial- NoDerivs 3.0 License located at http://creativecommons.org/licenses/by-nc-nd/4.0 and successor locations for such license (the "CC License") provided that, in each case, the Educational Content is specifically marked as being subject to the CC License.

Without limiting the generality of the terms of the CC License, the following are types of uses that Udacity expressly defines as falling outside of the definition of "non-commercial":

- (a) the sale or rental of (i) any part of the Educational Content, ((ii) any derivative works based at least in part on the Educational (Content, or (iii) any collective work that includes any part of the (Educational Content;
- (b) the sale of access or a link to any part of the Educational (Content without first obtaining informed consent from the buyer (that the buyer is aware that the Educational Content, or such part (thereof, is available at the Website free of charge;
- (c) providing training, support, or editorial services that use or (reference the Educational Content in exchange for a fee;
- (d) the sale of advertisements, sponsorships, or promotions placed (on the Educational Content, or any part thereof, or the sale of (advertisements, sponsorships, or promotions on any website or blog (containing any part of the Educational Material, including without (limitation any "pop-up advertisements";
- (e) the use of Educational Content by a college, university, school, or other educational institution for instruction where tuition is charged; and
- (f) the use of Educational Content by a for-profit corporation or non-profit entity for internal professional development or training.

## Table Of Contents

### Tutorials

### Introduction to Neural Networks

* [Introduction to Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.
* [Sentiment Analysis with NumPy](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
* [Introduction to PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch): Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

### Convolutional Neural Networks

* [Convolutional Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/convolutional-neural-networks): Visualize the output of layers that make up a CNN. Learn how to define and train a CNN for classifying [MNIST data](https://en.wikipedia.org/wiki/MNIST_database), a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
* [Transfer Learning](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/transfer-learning). In practice, most people don't train their own networks on huge datasets; they use **pre-trained** networks such as VGGnet. Here you'll use VGGnet to help classify images of flowers without training an end-to-end network from scratch.
* [Weight Initialization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/weight-initialization): Explore how initializing network weights affects performance.
* [Autoencoders](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder): Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.
* [Style Transfer](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer): Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!

### Recurrent Neural Networks

* [Intro to Recurrent Networks (Time series & Character-level RNN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how to implement these in PyTorch for a variety of tasks.
* [Embeddings (Word2Vec)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
* [Sentiment Analysis RNN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.
* [Attention](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/attention): Implement attention and apply it to annotation vectors.

### Generative Adversarial Networks

* [Generative Adversarial Network on MNIST](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist): Train a simple generative adversarial network on the MNIST dataset.
* [Batch Normalization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
* [Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
* [CycleGAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan): Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.

### Deploying a Model (with AWS SageMaker)

* [All exercise and project notebooks](https://github.com/udacity/sagemaker-deployment) for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.

### Projects

* [Predicting Bike-Sharing Patterns](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing): Implement a neural network in NumPy to predict bike rentals.
* [Dog Breed Classifier](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification): Build a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.
* [TV Script Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation): Train a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.
* [Face Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-face-generation): Use a DCGAN on the CelebA dataset to generate images of new and realistic human faces.

### Elective Material

* [Intro to TensorFlow](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/tensorflow/intro-to-tensorflow): Starting building neural networks with TensorFlow.
* [Keras](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/keras): Learn to build neural networks and convolutional neural networks with Keras.

---

# Dependencies
Inclusion of `numpy` and `matplotlib` libraries has been made for manipulating output in desired format.

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

If you'd like to learn more about version control and using `git` from the command line, take a look at our [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/deep-learning-v2-pytorch.git
cd deep-learning-v2-pytorch
```

2. Create (and activate) a new environment, named `deep-learning` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n deep-learning python=3.6
	source activate deep-learning
	```
	- __Windows__: 
	```
	conda create --name deep-learning python=3.6
	activate deep-learning
	```
	
	At this point your command line should look something like: `(deep-learning) <User>:deep-learning-v2-pytorch <user>$`. The `(deep-learning)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

7. That's it!

Now most of the `deep-learning` libraries are available to you. Very occasionally, you will see a repository with an addition requirements file, which exists should you want to use TensorFlow and Keras, for example. In this case, you're encouraged to install another library to your existing environment, or create a new environment for a specific project. 

Now, assuming your `deep-learning` environment is still activated, you can navigate to the main repo and start looking at the notebooks:

```
cd
cd deep-learning-v2-pytorch
jupyter notebook
```

To exit the environment when you have completed your work session, simply close the terminal window.
