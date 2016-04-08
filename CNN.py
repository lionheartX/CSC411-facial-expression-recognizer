#Hyperparameters
TRAIN_TEST_SPLIT_RATIOS = [0.1, 0.15, 0.2]
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
	X *= (1.0/X.max())
	print X.shape
	y = data['tr_labels'][:, 0]
	return X, y

def CNN(X, y):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state = 42)
	nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=7, kernel_shape=(3,3)),
        Layer("Softmax")], learning_rate=0.02, n_iter=10)
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
	print "Import libaries..."
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import datasets, cross_validation
	from sknn.mlp import Classifier, Convolution, Layer
	import logging
	logging.basicConfig()
	#digits = datasets.load_digits()
	#X = digits.images
	#y = digits.target
	#CNN(X, y)
	#print X
	#print X.shape
	print("-------------------------")
	main()
