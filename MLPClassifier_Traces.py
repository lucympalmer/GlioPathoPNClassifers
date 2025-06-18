# Multi-layer Perceptron Classifier using the traces of neurons

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

accuracy = np.zeros([100, 1])
for i in range(100):
    # Split into train and test data randomly
    indices = np.random.permutation(data.shape[0])
    allData = data[indices]
    grade1 = allData[allData[:, 0] == np.float64(1.0)]
    grade2 = allData[allData[:, 0] == np.float64(2.0)]
    rowNum = 37  # for the train data
    train_data = np.concatenate((grade1[0:rowNum + 1], grade2[0:rowNum + 1]), axis=0)
    test_data = np.concatenate((grade1[rowNum:], grade2[rowNum:]), axis=0)

    # Train and test datasets
    trainX = train_data[:, 1:]
    trainY = (train_data[:, 0] - 1).astype(int)
    testX = test_data[:, 1:]
    testY = (test_data[:, 0] - 1).astype(int)

    # Train MLPClassifier model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp_classifier', MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=None,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False
        ))
    ])

    model.fit(trainX, trainY)

    # Predict probabilities
    predProba = model.predict_proba(testX)

    # Predict class labels
    predClass = model.predict(testX)

    # Evaluate accuracy
    accuracy[i, 0] = accuracy_score(testY, predClass)
