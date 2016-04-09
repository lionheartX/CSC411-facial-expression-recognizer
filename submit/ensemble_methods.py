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


def EnsembleMethod(X, y):

	# divide our data set into a training set and a test set
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    								X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

	# train with decision tree classifier
	decisionTreeClassifier = DecisionTreeClassifier(max_depth=None, 
							   min_samples_split=1, random_state=0)

	# use the classifier to fit the data.
	decisionTreeClassifier.fit(X_train, y_train)

	# print the performance of the classifier
	print("====== Decision Tree Classifier ========")
	print('TRAIN SCORE', decisionTreeClassifier.score(X_train, y_train))
	print('TEST SCORE', decisionTreeClassifier.score(X_test, y_test)) 

	# train with random forest classifier
	randomForestClassifier = RandomForestClassifier(n_estimators=10,
					max_depth=None, min_samples_split=1, random_state=0)   

	# use the classifier to fit the data.
	randomForestClassifier.fit(X_train, y_train)

	# print the performance of the classifier
	print("====== Random Forest Classifier ========")
	print('TRAIN SCORE', randomForestClassifier.score(X_train, y_train))
	print('TEST SCORE', randomForestClassifier.score(X_test, y_test)) 

	# train with  extra trees classifier
	extraTreesClassifier = ExtraTreesClassifier(n_estimators=10,
				max_depth=None, min_samples_split=1, random_state=0)

	# use the classifier to fit the data.
	extraTreesClassifier.fit(X_train, y_train)

	# print the performance of the classifier
	print("======= Extra Trees Classifier ========")
	print('TRAIN SCORE', extraTreesClassifier.score(X_train, y_train))
	print('TEST SCORE', extraTreesClassifier.score(X_test, y_test)) 


def main():
    X, y = loadData()
    EnsembleMethod(X, y)


if __name__ == '__main__':
	print "Decision Tree and two Ensemble Methods:"
	from sklearn.cross_validation import cross_val_score
	from sklearn.datasets import make_blobs 
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.tree import DecisionTreeClassifier
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import linear_model, cross_validation
	print("-------------------------")
	main()
