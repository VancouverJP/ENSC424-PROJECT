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
from tensorflow.keras.applications.resnet50 import ResNet50

resnet_weights_path = 'res50_basemodel.h5'

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
def define_model_res_old(numBlock, useDropOut):
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


def define_model_res(numBlock, useDropOut):
	model = Sequential()

	# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
	# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
	model.add(ResNet50(include_top = False, pooling = 'avg', weights = resnet_weights_path))
	
	# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
	model.add(Dense(2, activation = 'softmax'))
	
	# Say not to train first layer (ResNet) model as it is already trained
	model.layers[0].trainable = False
	
	sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
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
	
def download_resnet50_pretrained_model():
	
	
	base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(200,200,3), pooling='avg')
	base_model.save("res50_basemodel.h5")
	
def main():
	define_model_res(0,0)
	
main()