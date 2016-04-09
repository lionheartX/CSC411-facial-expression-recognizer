#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.25


def loadData():
	# load the labeled images
    data = sio.loadmat('./labeled_images.mat')

    # reshape the data including images and the corresponding labels
    y_data = data['tr_labels'][:, 0];
    X_data = data['tr_images'].T;
    X_data = X_data.reshape((-1, 1024)).astype('float');

    return X_data, y_data


def SVM(X, y):

	# divide our data set into a training set and a test set
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

	classifier_poly2 = svm.SVC(kernel='poly', degree = 2)
	classifier_poly2.fit(X_train, y_train)
	print("======= poly degree=2 ========")
	print('TRAIN SCORE', classifier_poly2.score(X_train, y_train))
	print('TEST SCORE', classifier_poly2.score(X_test, y_test))

	# add reverse version to the training set
	for image in range(len(X_train)):
		X_reverse = np.fliplr(X_train[image].reshape(32, 32)).ravel()
		X_train = np.append(X_train, [X_reverse], axis=0)

	y_train = np.append(y_train, y_train)
	classifier = svm.SVC(kernel='poly', degree = 3, C = 10)
	classifier.fit(X_train, y_train)

	classifier_poly2_double = svm.SVC(kernel='poly', degree = 2)
	classifier_poly2_double.fit(X_train, y_train)
	print("======= poly degree=2 double ========")
	print('TRAIN SCORE', classifier_poly2_double.score(X_train, y_train))
	print('TEST SCORE', classifier_poly2_double.score(X_test, y_test))


def main():
	X, y = loadData()
	SVM(X, y)

if __name__ == '__main__':
	print "SVM Kernel Comparison:"
	import numpy as np
	import scipy.io as sio
	from sklearn import svm, cross_validation
	print("-------------------------")
	main()
