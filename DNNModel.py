import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam, Nadam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

# parameter dictionary
paramDict = {
    'filter': 512,
    'kernel': 8,
    'epoch': 100,
    'batchSize': 32,
    'dropOut': 0.3,
    'layer1': 1024,
    'layer2': 512,
    'layer3': 128,
    'learningRate': 0.01,
    'decay': 1e-6,
    'nesterov': True,
    'momentum': 0.9,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'activation1': 'relu',
    'activation2': 'sigmoid',
    'monitor': 'val_acc',  # param for checkpoint
    'verbose': 0,
    'save_best_only': True,
    'mode': 'max'
}

optimizerDict = {
    'sgd': SGD(paramDict['learningRate'], paramDict['nesterov'], paramDict['decay'], paramDict['momentum']),
    'rmsprop': RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
    'adadelta': Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
    'adam': Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    'nadam': Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
}


class BuildDNNModel(object):

    def __init__(self, data, bins, f_tp, f_fp, f_th):
        super(BuildDNNModel, self).__init__()
        self.data = data
        self.bins = bins

        self.corr_feats = data.getCorrFeats()
        self.feat_index_dict = dict()
        self.feat_importance_dict = dict()

        self.evaluationInfo = dict()

        self.trainingData = data.getTrainingData()
        self.validationData = data.getValidationData()
        self.testingData = data.getTestingData()

        X_train, Y_train = separateDataAndClassLabel(self.trainingData)
        X_valid, Y_valid = separateDataAndClassLabel(self.validationData)
        X_test, Y_test = separateDataAndClassLabel(self.testingData)

        X_train = data.getScaledData(X_train)
        X_valid = data.getScaledData(X_valid)
        X_test = data.getScaledData(X_test)

        self.numberOfClasses = encodeClassLabel(Y_train)
        self.numberOfFeatEachBin = math.floor(
            float(X_train.shape[1]) / bins)  # this is the number of attributed in each bin of matrix

        # reshaping class labels
        Y_train_reshaped = np_utils.to_categorical(Y_train, self.numberOfClasses)
        Y_valid_reshaped = np_utils.to_categorical(Y_valid, self.numberOfClasses)
        Y_test_reshaped = np_utils.to_categorical(Y_test, self.numberOfClasses)

        self.dataDict = {
            'train': X_train,
            'trainLabel': Y_train_reshaped,
            'valid': X_valid,
            'validLabel': Y_valid_reshaped,
            'test': X_test,
            'testLabel': Y_test_reshaped
        }

        self.evaluationInfo = buildAndRunModel(self.dataDict, int(self.numberOfFeatEachBin), self.bins,
                                                self.numberOfClasses, f_tp, f_fp, f_th)

    # returns a dictionary containing all evaluation statistics
    def getEvaluationStat(self):
        return self.evaluationInfo


# returns the TP, TN, FP and FN values
def getTPTNValues(test, testPred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(testPred)):
        if test[i] == testPred[i] == 1:
            TP += 1
        if testPred[i] == 1 and test[i] != testPred[i]:
            FP += 1
        if test[i] == testPred[i] == 0:
            TN += 1
        if testPred[i] == 0 and test[i] != testPred[i]:
            FN += 1

    return TP, TN, FP, FN


# separating feature matrix and class label
def separateDataAndClassLabel(dataMatrix):
    featureMatrix = dataMatrix[:, :(dataMatrix.shape[1] - 1)]
    classLabelMatrix = dataMatrix[:, -1]

    return featureMatrix, classLabelMatrix


# returns the number of classes and encode it
def encodeClassLabel(classLabel):
    labelEncoder = LabelEncoder().fit(classLabel)
    labels = labelEncoder.transform(classLabel)
    classes = list(labelEncoder.classes_)

    return len(classes)


# reshaping the data to the number of bins sizes
def reshapeDataToBinSize(dataMatrix, numberOfFeatEachBin, bins):
    ReshapedData = np.zeros((len(dataMatrix), numberOfFeatEachBin, bins))
    start = 0
    end = numberOfFeatEachBin

    for i in range(1, bins + 1):
        ReshapedData[:, :, i - 1] = dataMatrix[:, start:end * i]
        start = end * i
    return ReshapedData


# building the DNN model and run with the data, returns a list of metrics
def buildAndRunModel(dataDict, numberOfFeatEachBin, bins, numberOfClasses, f_tp, f_fp, f_th):
    trainData = dataDict['train']
    trainLabel = dataDict['trainLabel']
    validData = dataDict['valid']
    validLabel = dataDict['validLabel']
    testData = dataDict['test']
    testLabel = dataDict['testLabel']

    # building NN model
    model = Sequential()
    model.add(Dense(128, activation = paramDict['activation1'], input_shape = (numberOfFeatEachBin, )))
    model.add(Dropout(paramDict['dropOut']))
    model.add(Dense(256, activation = paramDict['activation1']))
    model.add(Dropout(paramDict['dropOut']))
    model.add(Dense(512, activation = paramDict['activation1']))
    model.add(Dropout(paramDict['dropOut']))
    model.add(Dense(1024, activation = paramDict['activation1']))
    model.add(Dropout(paramDict['dropOut']))
    model.add(Dense(1024, activation=paramDict['activation1']))
    model.add(Dropout(paramDict['dropOut']))
    model.add(Dense(1024, activation=paramDict['activation1']))
    model.add(Dropout(paramDict['dropOut']))
    model.add(Dense(numberOfClasses, activation=paramDict['activation2']))

    model.compile(optimizer=optimizerDict['adadelta'],
                  loss=paramDict['loss'],
                  metrics=paramDict['metrics'])

    # saving best model by validation accuracy
    filePath = 'weights.best.hdf5'
    checkpointer = ModelCheckpoint(filepath=filePath, verbose=0, monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    # fit the model to the training data and verify with validation data
    model.fit(trainData, trainLabel,
              epochs=paramDict['epoch'],
              callbacks=[checkpointer, earlystopper],
              batch_size=paramDict['batchSize'],
              shuffle=True,
              verbose=1,
              validation_data=(validData, validLabel))

    # load best model and compile
    model.load_weights('weights.best.hdf5')
    model.compile(optimizer=optimizerDict['adadelta'],
                  loss=paramDict['loss'],
                  metrics=paramDict['metrics'])

    # evaluation scores
    roc_auc = metrics.roc_auc_score(testLabel, model.predict(testData))
    precision = metrics.average_precision_score(testLabel, model.predict(testData))

    # get predicted class label
    probs = model.predict_proba(testData)
    testPredLabel = model.predict(testData)
    true_y = list()
    for y_i in range(len(testLabel)):
        true_y.append(testLabel[y_i][1])
    probs = probs[:, 1]

    fpr, tpr, threshold = metrics.roc_curve(true_y, probs)

    for i in range(len(fpr)):
        f_fp.write(str(fpr[i]) + '\t')
    f_fp.write('\n')

    for i in range(len(tpr)):
        f_tp.write(str(tpr[i]) + '\t')
    f_tp.write('\n')

    for i in range(len(threshold)):
        f_th.write(str(threshold[i]) + '\t')
    f_th.write('\n')

    # convert back class label from categorical to integer label
    testLabelRev = np.argmax(testLabel, axis=1)
    testPredLabelRev = np.argmax(testPredLabel, axis=1)

    # get TP, TN, FP, FN to calculate sensitivity, specificity, PPV and accuracy
    TP, TN, FP, FN = getTPTNValues(testLabelRev, testPredLabelRev)

    sensitivity = float(TP) / float(TP + FN)
    specificity = float(TN) / float(TN + FP)
    PPV = float(TP) / float(TP + FP)
    accuracy = float(TP + TN) / float(TP + FP + FN + TN)

    # dictionary to store evaluation stat
    evaluationInfo = {
        'roc_auc': roc_auc,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'PPV': PPV,
        'accuracy': accuracy,
        'batch_size': paramDict['batchSize'],
        'activation': paramDict['activation2'],
        'dropout': paramDict['dropOut']
    }

    return evaluationInfo
