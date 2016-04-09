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


def EnsembleMethods(X, y):

	# divide our data set into a training set and a test set
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    								X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

	# get randomized PCA model
	num_components = 120
	print("Extracting the top %d eigenfaces from %d faces"
          % (num_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=num_components, whiten=True).fit(X_train)

    # use the PCA model on our training set and test set.
	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")

	# get decision tree classifier
	decision_tree_classifier = DecisionTreeClassifier(max_depth=None, 
							   min_samples_split=1, random_state=0)

	# use decision tree classifier to fit the data.
	decision_tree_classifier.fit(X_train_pca, y_train)

	# print the performance of decision tree classifier
	print("====== Decision Tree Classifier ========")
	print('TRAIN SCORE', decision_tree_classifier.score(X_train_pca, y_train))
	print('TEST SCORE', decision_tree_classifier.score(X_test_pca, y_test))

	# get random forest classifier
	random_forest_classifier = RandomForestClassifier(n_estimators=10,
					max_depth=None, min_samples_split=1, random_state=0)

	# use random forest classifier to fit the data.
	random_forest_classifier.fit(X_train_pca, y_train)

	# print the performance of decision tree classifier
	print("====== Random Forest Classifier ========")
	print('TRAIN SCORE', random_forest_classifier.score(X_train_pca, y_train))
	print('TEST SCORE', random_forest_classifier.score(X_test_pca, y_test))                       

	# get extra trees classifier
	extra_trees_classifier = ExtraTreesClassifier(n_estimators=10,
				max_depth=None, min_samples_split=1, random_state=0)

	# use extra trees classifier to fit the data.
	extra_trees_classifier.fit(X_train_pca, y_train)

	# print the performance of decision tree classifier
	print("====== Extra Trees Classifier ========")
	print('TRAIN SCORE', extra_trees_classifier.score(X_train_pca, y_train))
	print('TEST SCORE', extra_trees_classifier.score(X_test_pca, y_test))  


def main():
    X, y = loadData()
    EnsembleMethods(X, y)


if __name__ == '__main__':
	print "Ensemble Methods:"
	from sklearn.cross_validation import cross_val_score
	from sklearn.datasets import make_blobs 
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.tree import DecisionTreeClassifier
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import cross_validation
	from sklearn.decomposition import RandomizedPCA
	print("-------------------------")
	main()
