#Hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2


def loadData():
    data = sio.loadmat('./labeled_images.mat')
    #test = sio.loadmat('./public_test_images.mat')
    (a_data, b_data, n_images_data) = data["tr_images"].shape
    #print(test)
    #(a_test, b_test, n_images_test) = test["public_test_images"].shape

    X_data = np.reshape(np.swapaxes(data["tr_images"], 0, 2), (n_images_data, a_data * b_data))
    #X_test = np.reshape(np.swapaxes(test["public_test_images"], 0, 2), (n_images_test, a_test * b_test))

    preprocessing.scale(X_data * 1.0, axis=1)
    #preprocessing.scale(X_test * 1.0, axis=1)
    y_data = np.reshape(data["tr_labels"], (n_images_data, ))
    return X, y


def SVM(X_train, y_train, X_test):

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)


# X_train = np.asarray(X_train)
# classifier = svm.SVC(kernel='precomputed')
# kernel_train = np.dot(X_train, X_train.T)  # linear kernel
# classifier.fit(kernel_train, y_train)
# print("-----------")

# Testing
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# kernel_test = np.dot(X_test, X_train.T)
# y_pred = classifier.predict(kernel_test)
# print("t_pred", y_pred)
# print("t_test", y_test)
# print accuracy_score(y_test, y_pred)
# print("======",1,"========")


# print('TRAIN SCORE', classifier.score(X_train, y_train))
# print('TEST SCORE', classifier.score(X_test, y_test))
# X_train_2 = X_train
# y_train_2 = y_train
# X_train_3 = X_train
    for image in range(len(X_train)):
        X_reverse = np.fliplr(X_train[image].reshape(32, 32)).ravel()
        X_train = np.append(X_train, [X_reverse], axis=0)

    y_train = np.append(y_train, y_train)
    classifier = svm.SVC(kernel='poly', degree = 2, C = 10)
    classifier.fit(X_train, y_train)
    # print("======",3,"========")
    print('TRAIN SCORE', classifier1.score(X_train, y_train))
    print('TEST SCORE', classifier1.score(X_test, y_test))
    return list(classifier.predict(X_test))
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
    result = SVM(X, y)
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
    logging.basicConfig()
    print("-------------------------")
    main()
