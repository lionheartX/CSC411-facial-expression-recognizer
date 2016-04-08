#Hyperparameters
TRAIN_TEST_SPLIT_RATIOS = [0.1, 0.15, 0.2]
ACTIVATION_FUNCTIONS = ["Rectifier", "", "Sigmoid"]
CHANNELS = [4, 8]
KERNEL_SHAPES=[(2,2),(3,3),(4,4),(16,1),(1,16)]

# dropout leasanier, GTX 970
# MNIST layer

def loadData():
	train = sio.loadmat('./labeled_images.mat')
	test = sio.loadmat('./public_test_images.mat')
	X_train = train['tr_images'].T
	X_train = X_train.astype('float')
	y_train = train['tr_labels'][:, 0]

	X_test = test['public_test_images'].T
	X_test = X_test.astype('float')
	#l2 normalize 
	preprocessing.normalize(X_train)
	preprocessing.normalize(X_test)
	#scale centre to the mean to unit vector
	#preprocessing.scale(X_train, =1)
	#preprocessing.scale(X_test)
	return X_train, y_train, X_test

def CNN(X_train, y_train, X_test):
	nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=20, kernel_shape=(5,5), dropout=0.25),
        Layer("Tanh", units=300),
        Layer("Tanh", units=100),
        Layer("Softmax")], learning_rate=0.02, n_iter=10)
	nn.fit(X_train, y_train)
	print('\nTRAIN SCORE', nn.score(X_train, y_train))
	return list(nn.predict(X_test))

def batch_run():
	pass

def output_to_csv(result):
	with open('pca1.csv', 'w') as f:
		f.write('Id,Prediction\n')
		index = 1
		for pred in result:
			f.write('%d,%d\n'%(index,pred))
			index +=1
		while index<=1253:
			f.write('%d,0\n'%(index))
			index +=1

def main():
	X_train, y_train, X_test = loadData()
	result = CNN(X_train, y_train, X_test)
	output_to_csv(result)

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
