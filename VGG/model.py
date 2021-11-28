# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:56:17 2021

@author: daiki
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation



#creating vgg model
def define_model_vgg(numBlock, useDropOut):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	if numBlock > 1:
		for i in range(1,numBlock):
			model.add(Conv2D(32 * (2 ** numBlock), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
			model.add(MaxPooling2D((2, 2)))
			
			if useDropOut:#add dropout of 20%
				model.add(Dropout(0.2))
                
        
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	
	if useDropOut:
		model.add(Dropout(0.5)) # dropbout of 50% after fully connected layer
	
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()
	return model

#create resnet model
def define_model_res(numBlock, useDropOut):
	# define model input
	visible = Input(shape=(200, 200, 3))
	# add residual module
	layer = residual_module(visible, 64)
	
	#flatten and dence to 1
	flatten = Flatten()(layer)
	dence1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
	output2 = Dense(1, activation='sigmoid')(dence1)
	
	# create model
	model = Model(inputs=visible, outputs=output2)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	# summarize model
	model.summary()
	
	return model

def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out



#create vgg or resnet model depending on the value of modelType
def define_model(modelType, numBlock, useDropOut):
	
	if modelType == 'vgg':
		return define_model_vgg(numBlock, useDropOut)
	elif modelType == 'res':
		return define_model_res(numBlock, useDropOut)
	else:
		print('No model defined')
		return