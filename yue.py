#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2


def loadData():
	data = sio.loadmat('./labeled_images.mat')
	
	(a, b, n_images) = data["tr_images"].shape

	X = np.reshape(np.swapaxes(data["tr_images"], 0, 2), (n_images, a * b))
	print X.shape
	preprocessing.scale(X * 1.0, axis=1)
	y = np.reshape(data["tr_labels"], (n_images, ))
	return X, y

def SVM(X, y):
def get_gabor_data():
    train_img, train_labels = loadData()
    n_images, _ = train_img.shape
    kernel = get_kernel()
    img = train_img[0]
    img = compute_feature(img.reshape(32, 32), kernel).ravel()
    result = np.ndarray((n_images, 32))
    for i in xrange(n_images):
    	print(i)
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

def SVM(X, y):

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)
	print(len(X_train))
	n_images, _ = X_train.shape
	classifier = svm.SVC(kernel='poly', degree = 2)
	print("okay!")
	scores = cross_validation.cross_val_score(classifier, X_train, np.reshape(y_train, (n_images, )) , cv=5)

	#classifier.fit(X_train, y_train)
	print("hi")
	print(scores)
	print("======",1,"========")
	# print('TRAIN SCORE', classifier.score(X_train, y_train))
	# print('TEST SCORE', classifier.score(X_test, y_test))



def main():
	X, y = get_gabor_data()
	print(X, y)
	SVM(X, y)

if __name__ == '__main__':
	print "Import libaries..."
	import numpy as np
	import scipy.io as sio
	import matplotlib.pyplot as plt
	from sklearn import svm, metrics, cross_validation
	import logging
	from skimage import data
	from sklearn import preprocessing
	from skimage.util import img_as_float
	from scipy import ndimage as ndi
	from skimage.filter import gabor_kernel
	logging.basicConfig()
	print("-------------------------")
	main()
