# baseline model for the dogs vs cats dataset
import sys

from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout # new in Dropout
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

#Explore Transfer Learning




#The architecture involves stacking convolutional layers with small 3×3 filters followed by a max pooling layer.
#Together, these layers form a block, and these blocks can be repeated where the number of filters in each block is increased 
#with the depth of the network such as 32, 64, 128, 256 for the first four blocks of the model. 
#Each layer will use the ReLU activation function and the He weight initialization, which are generally best practices. 
#a 3-block VGG-style architecture where each block has a single convolutional and pooling layer can be defined in Keras as follows:
# define cnn model
#def define_model():
#	model = sequential()
#	model.add(conv2d(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
#	model.add(maxpooling2d((2, 2)))
#	model.add(flatten())
#	model.add(dense(128, activation='relu', kernel_initializer='he_uniform'))
#	model.add(dense(1, activation='sigmoid'))
#	# compile model
#	opt = sgd(lr=0.001, momentum=0.9)
#	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#	return model

#two block vgg model, adds a second block with 64 filters.
# define cnn model
#def define_model():
#	model = Sequential()
#	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Flatten())
#	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#	model.add(Dense(1, activation='sigmoid'))
#	# compile model
#	opt = SGD(lr=0.001, momentum=0.9)
#	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#	return model

#The three-block VGG model extends the two block model and adds a third block with 128 filters.
# define cnn model
##don't have enough memeory to run the three-block VGG model 

#def define_model():
#	model = Sequential()
#	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Flatten())
#	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#	model.add(Dense(1, activation='sigmoid'))
#	# compile model
#	opt = SGD(lr=0.001, momentum=0.9)
#	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#	return model

#Dropout Regularization 
# define cnn model
#def define_model():
#	model = Sequential()
#	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Dropout(0.2))
#	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Dropout(0.2))
#	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#	model.add(MaxPooling2D((2, 2)))
#	model.add(Dropout(0.2))
#	model.add(Flatten())
#	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#	model.add(Dropout(0.5))
#	model.add(Dense(1, activation='sigmoid'))
#	# compile model
#	opt = SGD(lr=0.001, momentum=0.9)
#	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#	return model

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
#def run_test_harness():
#	# define model
#	model = define_model()
#	# create data generator
#    # scale the pixel values to the range of 0-1.
#	datagen = ImageDataGenerator(rescale=1.0/255.0)
#	# prepare iterators
#	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
#		class_mode='binary', batch_size=64, target_size=(200, 200))
#	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
#		class_mode='binary', batch_size=64, target_size=(200, 200))
#	# fit model
#    #The model will be fit for 20 epochs, a small number to check if the model can learn the problem.
#	history = model.fit(train_it, steps_per_epoch=len(train_it),
#		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
#	# evaluate model
#	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
#	print('> %.3f' % (acc * 100.0))
#	# learning curves
#	summarize_diagnostics(history)

# run the test harness for evaluating a model
#This requires that we have a separate ImageDataGenerator instance for the train and test dataset, then iterators for the train and 
#test sets created from the respective data generators.
#Image Data Augmentation


def run_test_harness():
	# define model
	model = define_model()
	# create data generators
    #Small changes to the input photos of dogs and cats might be useful for this problem, such as small shifts and horizontal flips
    #augmented with small (10%) random horizontal and vertical shifts and random horizontal flips that create a mirror image of a photo
	train_datagen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
