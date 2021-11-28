# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:36:17 2021

@author: daiki
"""

# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread


# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from os import makedirs
from numpy import asarray
from numpy import save
from random import random
from random import seed
from shutil import copyfile
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# load and confirm the shape
from numpy import load


# define location of dataset
folder = 'Dataset/dogs-vs-cats/train/train/'
dataset_home = 'dataset_dogs_vs_cats/'

#show some sample images of dogs
def showDogSamples():
    # plot first few images
    for i in range(9):
    	# define subplot
    	pyplot.subplot(330 + 1 + i)
    	# define filename
    	filename = folder + 'dog.' + str(i) + '.jpg'
    	# load image pixels
    	image = imread(filename)
    	# plot raw pixel data
    	pyplot.imshow(image)
    # show the figure
    pyplot.show()
    
    
#reshapes the image and saves as npy files 
def preprocessImg():
    photos, labels = list(), list()
    # enumerate files in the directory
    for file in listdir(folder):
    	# determine class
    	output = 0.0
    	if file.startswith('cat'):
    		output = 1.0
    	# load image
    	photo = load_img(folder + file, target_size=(200, 200))
    	# convert to numpy array
    	photo = img_to_array(photo)
    	# store
    	photos.append(photo)
    	labels.append(output)
    # convert to a numpy arrays
    photos = asarray(photos)
    labels = asarray(labels)
    print(photos.shape, labels.shape)
    # save the reshaped photos
    save('dogs_vs_cats_photos.npy', photos)
    save('dogs_vs_cats_labels.npy', labels)


def createDir():
    # create directories
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
    	# create label subdirectories
    	labeldirs = ['dogs/', 'cats/']
    	for labldir in labeldirs:
    		newdir = dataset_home + subdir + labldir
    		makedirs(newdir, exist_ok=True)

def copyImages():
    # seed random number generator
    seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25
    # copy training dataset images into subdirectories
    src_directory = folder
    for file in listdir(src_directory):
    	src = src_directory + '/' + file
    	dst_dir = 'train/'
    	if random() < val_ratio:
    		dst_dir = 'test/'
    	if file.startswith('cat'):
    		dst = dataset_home + dst_dir + 'cats/'  + file
    		copyfile(src, dst)
    	elif file.startswith('dog'):
    		dst = dataset_home + dst_dir + 'dogs/'  + file
    		copyfile(src, dst)


def loadImages():
    photos = load('dogs_vs_cats_photos.npy')
    labels = load('dogs_vs_cats_labels.npy')
    print(photos.shape, labels.shape)
    

copyImages()
