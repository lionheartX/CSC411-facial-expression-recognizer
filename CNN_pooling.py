ACTIVATION_FUNCTIONS = ["Rectifier", "", "Sigmoid"]
CHANNELS = [4, 8]
KERNEL_SHAPES=[(2,2),(3,3),(4,4),(16,1),(1,16)]
LAYERS = ["Softmax", ]

# dropout leasanier, GTX 970
# MNIST layer

def loadData():
	data = sio.loadmat('./labeled_images.mat')
	X = data['tr_images'].T
	X = X.astype('float')
	y = data['tr_labels'][:, 0]
	del data
	return X, y

def equalize_hist(train_img):
    n_images, _ = train_img.shape
    for i in xrange(n_images):
        img = train_img[i]
        train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
    return train_img

def CNN(X, y):
	#l2 normalize 
	#preprocessing.normalize(X, 'max')
	#scale centre to the mean to unit vector
	#preprocessing.scale(X_train)
	#preprocessing.scale(X_test)
	#X = equalize_hist(X)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state = 42)
	nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=100, kernel_shape=(10,10), dropout=0.25, 
        	normalize="batch", weight_decay=0.0001, pool_shape = (2,2), pool_type="max"),
        #Layer("Tanh", units=100),
        Layer("Softmax")], learning_rate=0.05, n_iter=10)
	nn.fit(X_train, y_train)
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
	# Use the GPU in 32-bit mode, falling back otherwise.
	from sknn.platform import gpu32
	import logging
	logging.basicConfig()
	print("-------------------------")
	main()
