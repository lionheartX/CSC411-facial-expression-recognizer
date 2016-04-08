#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.25


def loadData():
    data = sio.loadmat('./labeled_images.mat')
    #test = sio.loadmat('./public_test_images.mat')
    y_data = data['tr_labels'][:, 0];
    X_data = data['tr_images'].T;
    #print(test)
    #(a_test, b_test, n_images_test) = test["public_test_images"].shape

    X_data = X_data.reshape((-1, 1024)).astype('float');
    #X_test = np.reshape(np.swapaxes(test["public_test_images"], 0, 2), (n_images_test, a_test * b_test))
    # preprocessing.scale(X_data, axis=0)
    # preprocessing.normalize(X_data)
    #preprocessing.scale(X_test * 1.0, axis=1)
    return X_data, y_data

# def equalize_hist(train_img):
#     n_images, _ = train_img.shape
#     for i in xrange(n_images):
#         img = train_img[i]
#         train_img[i] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
#     return train_img


def SVM(X, y):

    X_train2, X_test, y_train2, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)


    # X_train = np.asarray(X_train)
    # classifier = svm.SVC(kernel='precomputed')
    # kernel_train = np.dot(X_train, X_train.T)  # linear kernel
    # classifier.fit(kernel_train, y_train)
    # print("-----------")

    # #Testing
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import confusion_matrix
    # kernel_test = np.dot(X_test, X_train.T)
    # y_pred = classifier.predict(kernel_test)
    # print("t_pred", y_pred)
    # print("t_test", y_test)
    # print accuracy_score(y_test, y_pred)
    # print("======",1,"========")

    # X_train2 = X_train
    # y_train2 = y_train
    # for image in range(len(X_train)):
    #     X_reverse = np.fliplr(X_train[image].reshape(32, 32)).ravel()
    #     X_train = np.append(X_train, [X_reverse], axis=0)

    # y_train = np.append(y_train, y_train)

    # classifier1 = svm.SVC(kernel='poly', degree = 2)
    # classifier1.fit(X_train, y_train)
    # print("====== poly 2 large ========")
    # print('TRAIN SCORE', classifier1.score(X_train, y_train))
    # print('TEST SCORE', classifier1.score(X_test, y_test))


    # classifier2 = svm.SVC(kernel='poly', degree = 3)
    # classifier2.fit(X_train, y_train)
    # print("====== poly 3 large ========")
    # print('TRAIN SCORE', classifier2.score(X_train, y_train))
    # print('TEST SCORE', classifier2.score(X_test, y_test))

    # classifier3 = svm.SVC(kernel='poly', degree = 3)
    # classifier3.fit(X_train2, y_train2)
    # print("====== poly 3 small ========")
    # print('TRAIN SCORE', classifier3.score(X_train2, y_train2))
    # print('TEST SCORE', classifier3.score(X_test, y_test))

    # X_train2 = equalize_hist(X_train2)

    # classifier1 = svm.SVC(kernel='poly', degree = 2)
    # classifier1.fit(X_train2, y_train2)
    # print("====== poly 2 small round 1 ========")
    # print('TRAIN SCORE', classifier1.score(X_train2, y_train2))
    # print('TEST SCORE', classifier1.score(X_test, y_test))


    preprocessing.robust_scale(X_train2, axis=1, with_centering = True)
    preprocessing.robust_scale(X_test, axis=1,  with_centering = True)
    preprocessing.normalize(X_train2)
    preprocessing.normalize(X_test)



    classifier2 = svm.SVC(kernel='poly', degree = 2)
    classifier2.fit(X_train2, y_train2)
    print("====== poly 2 small 2========")
    print('TRAIN SCORE', classifier2.score(X_train2, y_train2))
    print('TEST SCORE', classifier2.score(X_test, y_test))

    # X_train2 -= np.mean(X_train2, axis=1)[:, np.newaxis];
    # X_train2 /= np.sqrt(np.var(X_train2, axis=1) + 0.01)[:, np.newaxis];


    # classifier4 = svm.SVC(kernel='poly', degree = 2)
    # classifier4.fit(X_train2, y_train2)
    # print("====== poly 2 small 3 ========")
    # print('TRAIN SCORE', classifier4.score(X_train2, y_train2))
    # print('TEST SCORE', classifier4.score(X_test, y_test))


    # n_components = 10

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, X_train2.shape[0]))
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # X_train_pca = pca.transform(X_train2)
    # X_test_pca = pca.transform(X_test)
    # print("done ")


    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # classifier11 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # classifier11.fit(X_train_pca, y_train2)

    # print("====== PCA 10 ========")
    # print('TRAIN SCORE', classifier11.score(X_train_pca, y_train2))
    # print('TEST SCORE', classifier11.score(X_test_pca, y_test))

    

    # n_components = 50

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, X_train2.shape[0]))
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # X_train_pca = pca.transform(X_train2)
    # X_test_pca = pca.transform(X_test)
    # print("done ")


    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # classifier12 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # classifier12.fit(X_train_pca, y_train2)

    # print("====== PCA 50 ========")
    # print('TRAIN SCORE', classifier12.score(X_train_pca, y_train2))
    # print('TEST SCORE', classifier12.score(X_test_pca, y_test))




    

    # n_components = 100

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, X_train2.shape[0]))
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # X_train_pca = pca.transform(X_train2)
    # X_test_pca = pca.transform(X_test)
    # print("done ")


    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # classifier13.fit(X_train_pca, y_train2)

    # print("====== PCA 100 ========")
    # print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    # print('TEST SCORE', classifier13.score(X_test_pca, y_test))


    n_components = 120

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train2.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train2)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train2)

    print("====== PCA 120 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))

    n_components = 122

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train2.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train2)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train2)

    print("====== PCA 122 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))

    n_components = 130

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train2.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train2)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train2)

    print("====== PCA 130 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))

    # n_components = 135

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, X_train2.shape[0]))
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # X_train_pca = pca.transform(X_train2)
    # X_test_pca = pca.transform(X_test)
    # print("done ")


    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # classifier13.fit(X_train_pca, y_train2)

    # print("====== PCA 135 ========")
    # print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    # print('TEST SCORE', classifier13.score(X_test_pca, y_test))


    n_components = 147

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train2.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train2)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train2)

    print("====== PCA 147 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))

    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train2.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train2)
    X_test_pca = pca.transform(X_test)
    print("done ")



    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train2)

    print("====== PCA 150 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))



    n_components = 160

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train2.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train2)
    X_test_pca = pca.transform(X_test)
    print("done ")


    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier13.fit(X_train_pca, y_train2)

    print("====== PCA 160 ========")
    print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    print('TEST SCORE', classifier13.score(X_test_pca, y_test))
    # n_components = 200

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, X_train2.shape[0]))
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # X_train_pca = pca.transform(X_train2)
    # X_test_pca = pca.transform(X_test)
    # print("done ")


    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # classifier13.fit(X_train_pca, y_train2)

    # print("====== PCA 200 ========")
    # print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    # print('TEST SCORE', classifier13.score(X_test_pca, y_test))


    # n_components = 400

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, X_train2.shape[0]))
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train2)


    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # X_train_pca = pca.transform(X_train2)
    # X_test_pca = pca.transform(X_test)
    # print("done ")


    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # classifier13 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # classifier13.fit(X_train_pca, y_train2)

    # print("====== PCA 400 ========")
    # print('TRAIN SCORE', classifier13.score(X_train_pca, y_train2))
    # print('TEST SCORE', classifier13.score(X_test_pca, y_test))


    # classifier5 = svm.SVC(kernel='linear')
    # classifier5.fit(X_train2, y_train2)
    # print("====== linear ========")
    # print('TRAIN SCORE', classifier5.score(X_train2, y_train2))
    # print('TEST SCORE', classifier5.score(X_test, y_test))

   # return list(classifier.predict(X_test))
# classifier2 = svm.SVC(kernel='poly', degree = 2)
# classifier2.fit(X_train, y_train)
# print("======",2,"========")
# print('TRAIN SCORE', classifier2.score(X_train, y_train))
# print('TEST SCORE', classifier2.score(X_test, y_test))

# classifier1 = svm.SVC(kernel='poly', degree = 3)
# classifier1.fit(X_train_3, y_train)
# print("======",3,"========")
# print('TRAIN SCORE', classifier1.score(X_train_3, y_train))
# print('TEST SCORE', classifier1.score(X_test, y_test))
# classifier2 = svm.SVC(kernel='poly', degree = 2)
# classifier2.fit(X_train_3, y_train)
# print("======",2,"========")
# print('TRAIN SCORE', classifier2.score(X_train_3, y_train))
# print('TEST SCORE', classifier2.score(X_test, y_test))

# classifier1 = svm.SVC(kernel='poly', degree = 3)
# classifier1.fit(X_train_2, y_train_2)
# print("======",3,"========")
# print('TRAIN SCORE', classifier1.score(X_train_2, y_train_2))
# print('TEST SCORE', classifier1.score(X_test, y_test))
# classifier2 = svm.SVC(kernel='poly', degree = 2)
# classifier2.fit(X_train_2, y_train_2)
# print("======",2,"========")
# print('TRAIN SCORE', classifier2.score(X_train_2, y_train_2))
# print('TEST SCORE', classifier2.score(X_test, y_test))

# classifier3 = svm.SVC(kernel='poly', degree = 2, C = 100000000000)
# classifier3.fit(X_train, y_train)
# print("======",2,"========")
# print('TRAIN SCORE', classifier3.score(X_train, y_train))
# print('TEST SCORE', classifier3.score(X_test, y_test))



def main():
    X, y = loadData()
    SVM(X, y)
    # print(len(X_test))
    # with open('submit_normalized_poly1.csv', 'w') as f:
    #     f.write('Id,Prediction\n')
    #     index = 1
    #     for pred in result:
    #         f.write('%d,%d\n'%(index, pred))
    #         index += 1
    #     while index<=1253:
    #         f.write('%d,0\n'%(index))
    #         index+=1

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
    logging.basicConfig()
    print("-------------------------")
    main()
