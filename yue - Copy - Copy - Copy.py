#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2


def loadData():
    data = sio.loadmat('./labeled_images.mat')
    test = sio.loadmat('./public_test_images.mat')
    # (a_data, b_data, n_images_data) = data["tr_images"].shape
    # print(test)
    # (a_test, b_test, n_images_test) = test["public_test_images"].shape

  
    # ids = data['tr_identity']
    y_data = data['tr_labels'][:, 0];
    X_data = data['tr_images'].T;
    X_test = test["public_test_images"].T;
    # del data;
    # Preprocess images

    X_data = X_data.reshape((-1, 1024)).astype('float');
    X_test = X_test.reshape((-1, 1024)).astype('float');
    preprocessing.scale(X_test, axis=1)
    preprocessing.scale(X_data, axis=1)
    #preprocessing.scale(X_test * 1.0, axis=1)
    # y_data = np.reshape(data["tr_labels"], (n_images_data, ))
    return X_data, y_data, X_test 


def SVM(X_train, y_train, X_test):

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)


    for image in range(len(X_train)):
        X_reverse = np.fliplr(X_train[image].reshape(32, 32)).ravel()
        X_train = np.append(X_train, [X_reverse], axis=0)

    y_train = np.append(y_train, y_train)




    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 300

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier2 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier2.fit(X_train_pca, y_train)
    # print("======",3,"========")
    # print('TRAIN SCORE', classifier2.score(X_train_pca, y_train))
    # print('TEST SCORE', classifier2.score(X_test_pca, y_test))


    return list(classifier2.predict(X_test_pca))


def main():
    X_data, y_data, X_test = loadData()
    result = SVM(X_data, y_data, X_test)
    print(len(X_test))
    with open('pca1.csv', 'w') as f:
        f.write('Id,Prediction\n')
        index = 1
        for pred in result:
            f.write('%d,%d\n'%(index, pred))
            index += 1
        while index<=1253:
            f.write('%d,0\n'%(index))
            index+=1


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
    logging.basicConfig()
    print("-------------------------")
    main()
