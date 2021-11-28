# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 22:24:08 2021

@author: daiki
"""
# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from model import define_model

modelType = 'vgg' #vgg or res
numBlock = 4
useDropOut = True

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	pyplot.ylabel('loss')
	pyplot.xlabel('epoch')
	pyplot.legend(['train', 'test'], loc='upper left')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	pyplot.legend(['train', 'test'], loc='upper left')
	pyplot.ylabel('accuracy')
	pyplot.xlabel('epoch')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	batch_size = 64
	
	
	# define model
	model = define_model(modelType, numBlock, useDropOut)
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	if modelType == 'vgg':
		batch_size = 64
	elif modelType == 'res':
		batch_size = 32
		
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=batch_size, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=batch_size, target_size=(200, 200))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('=================== %.3f' % (acc * 100.0))
	
	# save model
	model.save('model.h5')
	
	# learning curves
	summarize_diagnostics(history)

def main():
    now = datetime.now()
    
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    # entry point, run the test harness
    run_test_harness()  
    
    
    endTime = datetime.now()
    
    endTimeDisplay = endTime.strftime("%H:%M:%S")
    print("End Time =", endTimeDisplay)


main()