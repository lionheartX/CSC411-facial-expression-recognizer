def double_rainbow():
    train_img, train_labels = get_training_data()
    return normalize(equalize_hist(train_img)), train_labels

def get_gabor_data():
    train_img, train_labels = get_training_data()
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

def normalize(train_img):
    return preprocessing.scale(train_img * 1.0, axis=1)

def equalize_hist(train_img):
    n_images, _ = train_img.shape
    for i in xrange(n_images):
        img = train_img[i]
        train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
    return train_img

def get_equalized_data():
    # Load an example image
    train_img, train_labels = get_training_data()
    return equalize_hist(train_img), train_labels 

def get_normalized_data():
    # Load an example image
    train_img, train_labels = get_training_data()
    return normalize(train_img), train_labels

def get_training_data():
    train = scipy.io.loadmat('labeled_images.mat')
    (x, y, n_images) = train["tr_images"].shape
    train_img = np.reshape(np.swapaxes(train["tr_images"], 0, 2), (n_images, x * y))
    return train_img, train['tr_labels']

def get_filtered_training_data(normalize=True):
    train = scipy.io.loadmat('labeled_images.mat')
    if normalize:
        filtered = scipy.io.loadmat('filtered_normalized.mat')
    else:
        filtered = scipy.io.loadmat('filtered_testimg.mat')
    print "Filtered training set: ", filtered['tr_images'].shape

    return filtered['tr_images'], train['tr_labels']

def train_SVM():
    train_img_original, train_labels = get_training_data()
    n_images, _ = train_img_original.shape
    train_img = np.ndarray((n_images, 32 * 20))
    
    train_parts = False
    if train_parts:
        for i in xrange(n_images):
            # train eyes only
            img = train_img_original[i]
            img = img.reshape(32, 32)
            img = img[:, 12:].ravel()
            train_img[i] = img
    else:
        train_img = train_img_original

    experiment = ["rbf", "sigmoid", "linear", "poly"]

    # iterate through all linear
    if "rbf" in experiment:
        for c in [10]:
            clf = svm.SVC(kernel='rbf', C=c) 
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
            print "rbf mean @C=", c, ": ", scores.mean()

    # iterate through all linear
    if "linear" in experiment:
        for c in [10]:
            clf = svm.SVC(kernel='linear', C=c)
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
            print "linear mean: ", scores.mean()

    # iterate through all poly
    if "poly" in experiment:
        for deg in xrange(1, 4):
            clf = svm.SVC(kernel='poly', degree=deg)
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
            print "POLY: ", deg,  " MAX: ", scores.max(), " MIN: ", scores.min(), " MEAN: ", scores.mean()

    # iterate through all sigmoid
    if "sigmoid" in experiment:
        clf = svm.SVC(kernel='sigmoid')
        scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
        print "Sigmoid mean: ", scores.mean()
    return clf

def classify(classifier, samples):
    return classifier.predict(samples)

def classify_pub_test(classifier):
    test = scipy.io.loadmat('public_test_images.mat')
    print test
    print test["public_test_images"].shape
    #return classifier.predict(samples)

def show_img(img):
    plt.imshow(img.reshape(32, 32))
    plt.show()

def testbench():
    cv2.getGaborKernel

def main():
    #preproc()
    #get_training_data()
    #get_equalized_data()
    classifier = train_SVM()
    #get_gabor_data()
    #classify_pub_test(None)

if __name__ == '__main__':
    print "Importing libs..."
    import scipy.io
    import pylab
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import svm
    from sklearn import cross_validation
    from sklearn import preprocessing
    from skimage import exposure
    from skimage import data
    print "done!"
    main()