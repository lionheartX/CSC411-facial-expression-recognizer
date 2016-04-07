#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2


def loadData():
	data = sio.loadmat('./labeled_images.mat')
	(x, y, n_images) = data["tr_images"].shape
	X = np.reshape(np.swapaxes(data["tr_images"], 0, 2), (n_images, x * y))
	#X = X.reshape((-1, 1024)).astype('float');
	#X *= (1.0/X.max())
	print X.shape
	y = np.reshape(data["tr_labels"], (n_images, ))
	return X, y

def SVM(X, y):

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

	classifier = svm.SVC(kernel='poly', degree=3)
	classifier.fit(X_train, y_train)


	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)
	# Create a classifier: a support vector classifier
	

	# We learn the digits on the first half of the digits
	#classifier.fit(X[:len(X) / 2], y[:len(X) / 2])
	#classifier.fit(X_train, y_train)

	# Now predict the value of the digit on the second half:
	# expected = X_test
	# predicted = classifier.predict(y_test)

	# print("Classification report for classifier %s:\n%s\n"
	#       % (classifier, metrics.classification_report(expected, predicted)))
	# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


	print('\nTRAIN SCORE', classifier.score(X_train, y_train))
	print('TEST SCORE', classifier.score(X_test, y_test))

def main():
	X, y = loadData()
	SVM(X, y)

if __name__ == '__main__':
	print "Import libaries..."
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import svm, metrics, cross_validation
	import logging
	logging.basicConfig()
	print("-------------------------")
	main()
