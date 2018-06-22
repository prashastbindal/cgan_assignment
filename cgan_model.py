
from keras.layers import Dense, Dropout, Concatenate, MaxoutDense, Input, Embedding, Flatten, Reshape
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt

import shutil
shutil.rmtree('logs')
shutil.rmtree('result')

import pathlib
pathlib.Path('logs').mkdir(exist_ok=True) 
pathlib.Path('result').mkdir(exist_ok=True) 


	
num_digits = 10
image_dim = (28, 28)


np.random.seed(1234)


def get_generator_model(z_size, y_size):

	z = Input(shape = (z_size,))

	y = Input(shape = (y_size,))
	y_embedding = Embedding(input_dim = num_digits, output_dim = z_size)(y)          
	y_embedding = Flatten()(y_embedding)


	z_out = Dense(200, activation='linear',  kernel_initializer='glorot_normal')(z)
	z_out = LeakyReLU()(z_out)
	z_out = Dropout(0.5)(z_out)

	y_out = Dense(1000, activation='linear',  kernel_initializer='glorot_normal')(y_embedding)
	y_out = LeakyReLU()(y_out)
	y_out = Dropout(0.5)(y_out)

	merged = Concatenate()([z_out, y_out])          

	out = Dense(1200, activation='linear',  kernel_initializer='glorot_normal')(merged)
	out = LeakyReLU()(out)
	out = Dropout(0.5)(out)

	out = Dense(np.prod(image_dim), activation = 'sigmoid')(out)           

	gen_image = Reshape(image_dim)(out)

	# model = Model([z, y], gen_image)
	# model.summary()

	return Model([z, y], gen_image)


def get_discriminator_model(y_size):

	image = Input(shape=image_dim)
	x = Flatten()(image)

	y = Input(shape = (y_size,))
	y_embedding = Embedding(input_dim = num_digits, output_dim = np.prod(image_dim))(y)		
	y_embedding = Flatten()(y_embedding)


	x_out = MaxoutDense(240, nb_feature=5)(x)
	x_out = Dropout(0.5)(x_out)

	y_out = MaxoutDense(50, nb_feature=5)(y_embedding)
	y_out = Dropout(0.5)(y_out)

	merged = Concatenate()([x_out, y_out])												

	out = MaxoutDense(240, nb_feature=4)(merged)
	out = Dropout(0.5)(out)

	out = Dense(1, activation = 'sigmoid',  kernel_initializer='glorot_normal')(out)

	# model = Model([image, y], out)
	# model.summary()

	return Model([image, y], out)


def get_cgan_model(generator_model, discriminator_model, z_size, y_size):
	# global discriminator_model
	# global generator_model

	z = Input(shape = (z_size,))
	y = Input(shape = (y_size,))

	gen_image = generator_model([z, y])

	discriminator_model.trainable = False

	is_real_image = discriminator_model([gen_image, y])

	model = Model([z, y], is_real_image)
	model.summary()

	return Model([z, y], is_real_image)




def get_plot_figure(nos_distribution, gen_images, digits):
	figure, axes = plt.subplots(nos_distribution[0], nos_distribution[1])

	k = 0
	for i in range(nos_distribution[0]):
		for j in range(nos_distribution[1]):
			axes[i,j].imshow(gen_images[k,:,:], cmap='gray')                  
			axes[i,j].axis('off')
			axes[i,j].set_title("%d" % digits[k])
			k += 1

	return figure


def save_gen_images(epoch, generator_model, z_size, y_size):
	# global generator_model
	
	sampled_y = np.arange(0, 10).reshape(10, y_size)

	nos_distribution = (2,5)
	z = np.random.normal(0, 1, (np.prod(nos_distribution), z_size))

	gen_images = generator_model.predict([z, sampled_y])

	figure = get_plot_figure(nos_distribution, gen_images, sampled_y)

	figure.savefig("result/%d.png" % epoch)
	plt.close()



def train_CGAN( batch_size, num_epochs):
	
	z_size = 100
	y_size = 1
	discr_training_iter = 1


	generator_model = get_generator_model(z_size, y_size)

	discriminator_model = get_discriminator_model(y_size)
	optimizer = Adam(0.0002, 0.5)
	discriminator_model.compile(loss=['binary_crossentropy'], optimizer=optimizer)
	
	cgan_model = get_cgan_model(generator_model, discriminator_model, z_size, y_size)
	cgan_model.compile(loss=['binary_crossentropy'], optimizer=optimizer)

	writer = tf.summary.FileWriter("logs/", max_queue=10)

	(images_train, y_train), (images_test, y_test) = mnist.load_data()
	
	images_train = images_train.astype(np.float32) / 255              

	y_train = np.reshape(y_train, (y_train.shape[0], y_size))

	for epoch in range(num_epochs):


		# label smoothing: instead of taking the binary values of 1 and 0 for real and fake images, it is better to label images with random numbers from a range
		is_real_batch = np.random.normal(0.7, 1.2, (batch_size, y_size))
		is_fake_batch = np.random.normal(0.0, 0.3, (batch_size, y_size))

		z_batch = np.random.normal(0, 1, (batch_size, z_size))

		randomIndices = np.random.randint(0, images_train.shape[0], batch_size)         
		images_train_batch = images_train[randomIndices]
		y_train_batch = y_train[randomIndices]

		gen_images_batch = generator_model.predict([z_batch, y_train_batch])


		for i in range(discr_training_iter):	
			discriminator_loss_real = discriminator_model.train_on_batch([images_train_batch, y_train_batch], is_real_batch)
			discriminator_loss_fake = discriminator_model.train_on_batch([gen_images_batch, y_train_batch], is_fake_batch)
			
		discriminator_loss = np.add(discriminator_loss_fake, discriminator_loss_real)

		summary = tf.Summary(value=[tf.Summary.Value(tag="discrim_loss", simple_value=discriminator_loss), ])
		writer.add_summary(summary)

		sampled_y = np.random.randint(0, 10, batch_size).reshape(batch_size, y_size)

		generator_loss = cgan_model.train_on_batch([z_batch, sampled_y], is_real_batch)

		summary = tf.Summary(value=[tf.Summary.Value(tag="generator_loss", simple_value=generator_loss), ])
		writer.add_summary(summary)


		print ("Epoch %d Generator loss: %f Discrim loss: %f" % (epoch, generator_loss, discriminator_loss))

		if epoch % 100 == 0:
			writer.flush()
			save_gen_images(epoch, generator_model, z_size, y_size)
			



if __name__ == '__main__':
    train_CGAN(100, 50000)









