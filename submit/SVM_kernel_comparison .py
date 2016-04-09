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

	classifier_linear = svm.SVC(kernel='linear')
	classifier_linear.fit(X_train, y_train)
	print("======= linear =======")
	print('TRAIN SCORE', classifier_linear.score(X_train, y_train))
	print('TEST SCORE', classifier_linear.score(X_test, y_test))

	classifier_poly2 = svm.SVC(kernel='poly', degree = 2)
	classifier_poly2.fit(X_train, y_train)
	print("======= poly degree=2 ========")
	print('TRAIN SCORE', classifier_poly2.score(X_train, y_train))
	print('TEST SCORE', classifier_poly2.score(X_test, y_test))

	classifier_poly3 = svm.SVC(kernel='poly', degree = 3)
	classifier_poly3.fit(X_train, y_train)
	print("======= poly degree=3 ========")
	print('TRAIN SCORE', classifier_poly3.score(X_train, y_train))
	print('TEST SCORE', classifier_poly3.score(X_test, y_test))

	classifier_poly4 = svm.SVC(kernel='poly', degree = 4)
	classifier_poly4.fit(X_train, y_train)
	print("======= poly degree=4 ========")
	print('TRAIN SCORE', classifier_poly4.score(X_train, y_train))
	print('TEST SCORE', classifier_poly4.score(X_test, y_test))

	classifier_poly5 = svm.SVC(kernel='poly', degree = 5)
	classifier_poly5.fit(X_train, y_train)
	print("======= poly degree=5 ========")
	print('TRAIN SCORE', classifier_poly5.score(X_train, y_train))
	print('TEST SCORE', classifier_poly5.score(X_test, y_test))

	classifier_rbf = svm.SVC(kernel='rbf')
	classifier_rbf.fit(X_train, y_train)
	print("======= rbf ========")
	print('TRAIN SCORE', classifier_rbf.score(X_train, y_train))
	print('TEST SCORE', classifier_rbf.score(X_test, y_test))

	classifier_sigmoid = svm.SVC(kernel='sigmoid')
	classifier_sigmoid.fit(X_train, y_train)
	print("======= sigmoid ========")
	print('TRAIN SCORE', classifier_sigmoid.score(X_train, y_train))
	print('TEST SCORE', classifier_sigmoid.score(X_test, y_test))


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
