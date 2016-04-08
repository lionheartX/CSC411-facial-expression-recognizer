#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2


def loadData():
    data = sio.loadmat('./labeled_images.mat')
    y_data = data['tr_labels'][:, 0];
    X_data = data['tr_images'].T;
    #print(test)
    #(a_test, b_test, n_images_test) = test["public_test_images"].shape
    del data
    X_data = X_data.reshape((-1, 1024)).astype('float');
    return X_data, y_data

# def equalize_hist(train_img):
#     n_images, _ = train_img.shape
#     for i in xrange(n_images):
#         img = train_img[i]
#         train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
#     return train_img

def get_equalized_data(X, y):
    # Load an example image
    return equalize_hist(X), y

def equalize_hist(train_img):
    n_images, _ = train_img.shape
    for i in xrange(n_images):
        img = train_img[i]
        train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
    return train_img

'''
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
'''


def SVM(X, y):
    print("SVM with PCA of rbf, writening all on, no normalize")
    preprocessing.normalize(X, 'max')
    #preprocessing.robust_scale(X, axis=1, with_centering = True) #bad
    X = equalize_hist(X)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

    n_components = 120

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=False).fit(X_train)


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

    n_components = 130

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=False).fit(X_train)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train)

    print("====== PCA 130 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))

    n_components = 147

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=False).fit(X_train)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train)

    print("====== PCA 147 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))

    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=False).fit(X_train)


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



    n_components = 160

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=False).fit(X_train)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train)

    print("====== PCA 160 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))



def write_to_csv(result):
    with open('submit_normalized_poly2.csv', 'w') as f:
        f.write('Id,Prediction\n')
        index = 1
        for pred in result:
            f.write('%d,%d\n'%(index, pred))
            index += 1
        while index<=1253:
            f.write('%d,0\n'%(index))
            index+=1    

def main():
    X, y= loadData()
    SVM(X, y)
    #write_to_csv(result)

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
    from sklearn.grid_search import GridSearchCV
    from sklearn.decomposition import RandomizedPCA
    from sklearn.svm import SVC
    from skimage import exposure
    # Use the maximum number of threads for this script.
    from sknn.platform import cpu32, threading
    logging.basicConfig()
    print("-------------------------")
    main()
