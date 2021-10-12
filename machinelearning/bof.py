import cv2
import numpy as np
import os
import imutils
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

train_path = "./dataset-classif/train/"
training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

des_list = []

sift = cv2.xfeatures2d.SIFT_create()
for image_path in image_paths:
    im = cv2.imread(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    des_list.append((image_path, des))

    cv2.drawKeypoints(gray, kp, im)

# BFMatcher avec les parametres par defauts
bf = cv2.BFMatcher()
match = bf.knnMatch(des_list[0], des_list[i], k=2)
matches.append(match)

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

k = 100
voc, variance = kmeans(descriptors, k, 1)

im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# transform vector
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(
    np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), "float32"
)
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

kn = 5
# clf = neighbors.KNeighborsClassifier()
clf = neighbors.KNeighborsClassifier(kn, weights="uniform", p=2, metric="minkowski")
# Linear SVM
# clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)

clf, classes_names, stdSlr, k, voc = joblib.load("bof2.pkl")
predictions = [classes_names[i] for i in clf.predict(test_features)]

im_features_r = preprocessing.normalize(im_features_r, norm="l2")

score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)
