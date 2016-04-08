#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2


def loadData():
	data = sio.loadmat('./labeled_images.mat')
	
	'''(a, b, n_images) = data["tr_images"].shape

	X = np.reshape(np.swapaxes(data["tr_images"], 0, 2), (n_images, a * b))
	print ("aaaaaaaaa",X.shape)
	preprocessing.scale(X * 1.0, axis=1)
	y = np.reshape(data["tr_labels"], (n_images, ))
	print(X, y, "yue")
	return X, y
	'''
	ids = data['tr_identity']
	y = data['tr_labels'][:, 0];
	X = data['tr_images'].T;
	del data;
	# Preprocess images

	X = X.reshape((-1, 1024)).astype('float');
	#X -= np.mean(X, axis=1)[:, np.newaxis];
	#X /= np.sqrt(np.var(X, axis=1) + 0.01)[:, np.newaxis];
	# Make ids unique (for -1s)
	#inc = itertools.count(start=-1, step=-1).__iter__();
	#uids = [id[0] if id != -1 else inc.next() for id in ids];
	return X, y

def pre_process(X, y, uids):
	pass


def get_gabor_data():
    train_img, train_labels = loadData()
    n_images, _ = train_img.shape
    kernel = get_kernel()
    img = train_img[0]
    img = compute_feature(img.reshape(32, 32), kernel).ravel()
    result = np.ndarray((n_images, 32))
    for i in xrange(n_images):
        img = train_img[i]
        result[i] = compute_feature(img.reshape(32, 32), kernel).ravel()
    return result, train_labels

def get_kernel():
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

def compute_feature(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def equalize_hist(X):
    num_samples, _ = X.shape
    for i in xrange(num_samples):
        sample = X[i]
        X[i] = exposure.equalize_hist(sample.reshape(32, 32)).ravel()
    return X

def SVM(X, y):

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)
	print(len(X_train))

    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
	n_components = 150
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


	print("Projecting the input data on the eigenfaces orthonormal basis")
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done ")

	X_train_pca = equalize_hist(X_train_pca)
	preprocessing.scale(X_train_pca * 1.0, axis=1)
	X_test_pca = equalize_hist(X_test_pca)
	preprocessing.scale(X_test_pca * 1.0, axis=1)

    # classifier = svm.SVC(kernel='poly', degree = 3)
    # classifier.fit(X_train, y_train)
    # # print("======",3,"========")
    # print('TRAIN SCORE', classifier.score(X_train, y_train))
    # print('TEST SCORE', classifier.score(X_test, y_test))


	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	classifier2 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	classifier2.fit(X_train_pca, y_train)
	# print("======",3,"========")
	print('TRAIN SCORE', classifier2.score(X_train_pca, y_train))
	print('TEST SCORE', classifier2.score(X_test_pca, y_test))


	# n_images, _ = X_train.shape
	# classifier = svm.SVC(kernel='poly', degree = 2)
	# print("okay!")
	# #scores = cross_validation.cross_val_score(classifier, X_train, np.reshape(y_train, (n_images, )) , cv=5)

	# classifier.fit(X_train, y_train)
	# print("hi")
	# # print(scores)
	# print("======",1,"========")
	# print('TRAIN SCORE', classifier.score(X_train, y_train))
	# print('TEST SCORE', classifier.score(X_test, y_test))



def main():
	X, y = get_gabor_data()
	print(X.shape)
	print(X, y)
	SVM(X, y)

if __name__ == '__main__':
	print "Import libaries..."
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import svm, metrics, cross_validation
	import logging
	from skimage import exposure
	from skimage import data
	from sklearn import preprocessing
	from skimage.util import img_as_float
	from scipy import ndimage as ndi
	from skimage.filters import gabor_kernel
	from sklearn.grid_search import GridSearchCV
	from sklearn.decomposition import RandomizedPCA
	from sklearn.svm import SVC
	logging.basicConfig()
	print("-------------------------")
	main()
