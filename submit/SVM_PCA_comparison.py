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

	n_components = 10

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")

	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier11 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier11.fit(X_train_pca, y_train)

	print("====== PCA 10 ========")
	print('TRAIN SCORE', classifier11.score(X_train_pca, y_train))
	print('TEST SCORE', classifier11.score(X_test_pca, y_test))

	

	n_components = 50

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier12 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier12.fit(X_train_pca, y_train)

	print("====== PCA 50 ========")
	print('TRAIN SCORE', classifier12.score(X_train_pca, y_train))
	print('TEST SCORE', classifier12.score(X_test_pca, y_test))
	

	n_components = 100

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 100 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


	n_components = 120

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
				'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 120 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


	n_components = 135

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 135 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


	n_components = 150

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")



	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 150 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


	n_components = 165

	print("Extracting the top %d eigenfaces from %d faces"
			% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 165 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


	n_components = 180

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 180 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))

	n_components = 200

	print("Extracting the top %d eigenfaces from %d faces"
		  % (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 200 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


	n_components = 400

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_train.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier13.fit(X_train_pca, y_train)

	print("====== PCA 400 ========")
	print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
	print('TEST SCORE', classifier13.score(X_test_pca, y_test))


def main():
	X, y = loadData()
	SVM(X, y)

if __name__ == '__main__':
	print "SVM PCA Comparisons:"
	import numpy as np
	import scipy.io as sio
	from sklearn import svm, cross_validation
	from sklearn.grid_search import GridSearchCV
	from sklearn.decomposition import RandomizedPCA
	from sklearn.svm import SVC
	print("-------------------------")
	main()
