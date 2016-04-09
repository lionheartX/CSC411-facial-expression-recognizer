'''
NN: a working 1-layer tanh facial expression recognition network
'''

def loadData():
	data = sio.loadmat('./labeled_images.mat')
	y = data['tr_labels'][:, 0]
	X = data['tr_images'].T
	X = X.reshape((-1, 1024)).astype('float')
	del data
	return X, y

def equalize_hist(train_img):
	n_images, _ = train_img.shape
	for i in xrange(n_images):
		img = train_img[i]
		train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
	return train_img

def CNN(X, y):
	print("1-layer Tanh 100 NN")
	#l2 normalize 
	preprocessing.normalize(X, 'max')
	print("Done normalization")
	X = equalize_hist(X)
	#print("Done histogram equalization")
	#scale centre to the mean to unit vector
	#preprocessing.scale(X_train)
	#preprocessing.scale(X_test)
	#X = equalize_hist(X)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
	print("Creating neural net...")
	nn = Classifier(
    layers=[
    	Layer("Tanh", units = 98, weight_decay=0.0001),
        Layer("Softmax")], learning_rate=0.01, n_iter=1000, batch_size= 5)
	print("Done creating neural net")
	print("Neural net fitting....")
	nn.fit(X_train, y_train)
	print("Done Neural net fitting!")
	print('\nTRAIN SCORE', nn.score(X_train, y_train))
	print('TEST SCORE', nn.score(X_test, y_test))
	#y_pred = nn.predict(X_test)

def batch_run():
	pass

def main():
	X, y = loadData()
	CNN(X, y)

if __name__ == '__main__':
	print "Code running ..."
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import preprocessing
	from sklearn import datasets, cross_validation
	from sknn.mlp import Classifier, Convolution, Layer

	from sknn.platform import cpu
	from skimage import exposure
	import logging
	logging.basicConfig()
	print("-------------------------")
	main()
