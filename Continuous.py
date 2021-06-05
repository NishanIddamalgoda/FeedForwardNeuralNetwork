from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy
import keras
from sklearn.metrics import f1_score

    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
    
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataset = numpy.loadtxt("crx.csv", delimiter=",")

# split into input (X) and output (Y) variables
# input variables (0-6 columns)
X = dataset[:,0:6]

# output variable (6 column)
Y = dataset[:,6]

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=6, activation='sigmoid'))    
    model.add(Dense(3, activation='sigmoid'))
    
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model  mean_squared_error binary_crossentropy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
    
    # Fit the model
    model.fit(X[train], Y[train], epochs=100, batch_size=10, verbose=1)
        
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
print (cvscores)
print (numpy.mean(cvscores))