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

def LogisticRegression(X, y):

	# divide our data set into a training set and a test set
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    									X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

	h = .02  # step size in the mesh

	logistic_regression = linear_model.LogisticRegression(C=1e5)

	# we create an instance of Neighbours Classifier and fit the data.
	logistic_regression.fit(X_train, y_train)

	# print the performance of logistic regression 
	print("====== Logistic Regression ========")
	print('TRAIN SCORE', logistic_regression.score(X_train, y_train))
	print('TEST SCORE', logistic_regression.score(X_test, y_test))

def main():
    X, y = loadData()
    LogisticRegression(X, y)

if __name__ == '__main__':
	print "Logistic Regression:"
	import numpy as np
	import scipy.io as sio
	from sklearn import linear_model, cross_validation
	print("-------------------------")
	main()





