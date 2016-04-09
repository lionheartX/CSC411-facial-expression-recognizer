#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.25


def loadData():
	# load the labeled images
	data = sio.loadmat('./labeled_images.mat')

	# reshape the data including images and the corresponding labels
	y_data = data['tr_labels'][:, 0];
	X_data = data['tr_images'].T;
	X_data = X_data.reshape((-1, 1024)).astype('float');
	test = sio.loadmat('./public_test_images.mat')
	hidden = sio.loadmat('./hidden_test_images.mat')
	X_test = test['public_test_images'].T;
	X_test = X_test.reshape((-1, 1024)).astype('float')
	X_hidden = hidden['hidden_test_images'].T;
	X_hidden = X_hidden.reshape((-1, 1024)).astype('float');

	return X_data, y_data, X_test, X_hidden

def get_equalized_data(X, y):
	# Load an example image
	return equalize_hist(X), y

def equalize_hist(train_img):
	n_images, _ = train_img.shape
	for i in xrange(n_images):
		img = train_img[i]
		train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
	return train_img

def SVM(X_data, y_data, X_test, X_hidden):

	X_data = equalize_hist(X_data) 
	preprocessing.normalize(X_data, 'max')
	preprocessing.scale(X_data, axis=1)

	X_test = equalize_hist(X_test) 
	preprocessing.normalize(X_test, 'max')
	preprocessing.scale(X_test, axis=1)

	X_hidden = equalize_hist(X_hidden) 
	preprocessing.normalize(X_hidden, 'max')
	preprocessing.scale(X_hidden, axis=1)


	# divide our data set into a training set and a test set
	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

	n_components = 150

	print("Extracting the top %d eigenfaces from %d faces"
		% (n_components, X_data.shape[0]))
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_data)

	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_data)
	X_test_pca = pca.transform(X_test)
	X_hidden_pca = pca.transform(X_hidden)
	print("done ")

	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier.fit(X_train_pca, y_data)

	pub_res = list(classifier.predict(X_test_pca))
	hid_res = list(classifier.predict(X_hidden_pca))

	return pub_res+hid_res

	# print("====== PCA 150 ========")
	# print('TRAIN SCORE', classifier11.score(X_train_pca, y_train))
	# print('TEST SCORE', classifier11.score(X_test_pca, y_test))



def main():
	X_data, y_data, X_test, X_hidden = loadData()
	print "Training SVM..."
	cls_res_list = SVM(X_data, y_data, X_test, X_hidden)
	print cls_res_list
	with open('submit PCA 150_11.csv', 'w') as f:
		f.write('Id,Prediction\n')
		index = 1
		for pred in cls_res_list:
			f.write('%d,%d\n'%(index, pred))
			index += 1
		while index<=1253:
			f.write('%d,0\n'%(index))
			index+=1


if __name__ == '__main__':
	print "SVM Best:"
	import numpy as np
	import scipy.io as sio
	from sklearn import svm, cross_validation
	from sklearn import preprocessing
	from sklearn.grid_search import GridSearchCV
	from sklearn.decomposition import RandomizedPCA
	from skimage import exposure
	from sklearn.svm import SVC
	print("-------------------------")
	main()
