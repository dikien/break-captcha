
# coding: utf-8

# In[7]:

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sknn.mlp import Classifier, Layer
from sknn.mlp import Classifier, Convolution, FastVectorSpace, Layer, MultiLayerPerceptron
import numpy as np
from time import time
from glob import glob
import os


# In[3]:

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


# In[4]:

### Plan2 ###


# In[17]:

features = joblib.load("./mldata/features_1200.mat")
labels = joblib.load("./mldata/lables_1200.mat")

features = np.array(features, 'int16')
labels = np.array(labels, 'int')

t0 = time()
def scale(X, eps = 0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

features = features.astype("float32")
features = scale(features)

print "escape time : ", round(time()-t0, 3), "s"

# scale the data to the range [0, 1] and then construct the training
# and testing splits
(trainX, testX, trainY, testY) = train_test_split(features, labels, test_size = 0.1)
print "the shape of training set %s rows, %s columns" %(trainX.shape[0], trainX.shape[1])
print "the shape of test set %s rows, %s columns" %(testX.shape[0], testX.shape[1])
print "the range of training set : %s ~ %s" %(trainX.min(),trainX.max())
print "the range of test set : %s ~ %s" %(testX.min(),testX.max())

# reshape for convolutions
trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
testX = testX.reshape((testX.shape[0], 1, 28, 28))


# In[ ]:

# Convolution 
nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=9, kernel_shape=(3,3), border_mode='full'),
        Convolution("Rectifier", channels=9, kernel_shape=(3,3), border_mode='full'),
        Convolution("Rectifier", channels=9, kernel_shape=(3,3), border_mode='full'),
        Convolution("Rectifier", channels=9, kernel_shape=(3,3), border_mode='full'),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=10,
    verbose=True)
nn.fit(trainX, trainY)

# compute the predictions for the test data and show a classification report
preds = nn.predict(testX)

print "accuracy score : %s" %(accuracy_score(testY, preds))
print "classification report : "
print classification_report(testY, preds)

