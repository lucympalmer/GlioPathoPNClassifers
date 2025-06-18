# Logistic Regression Classifier using the extracted features or traces of neurons

import numpy as np

from sklearn.linear_model import LogisticRegression
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
    trainData = np.concatenate((grade1[0:rowNum + 1], grade2[0:rowNum + 1]), axis=0)
    testData = np.concatenate((grade1[rowNum:], grade2[rowNum:]), axis=0)

    # Train and test datasets
    trainX = trainData[:, 1:]
    trainY = (trainData[:, 0] - 1).astype(int)
    testX = testData[:, 1:]
    testY = (testData[:, 0] - 1).astype(int)

    # Train Logistic Regression model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(
            solver='lbfgs',     # 'lbfgs' for scaled data
            max_iter=1000,
            random_state=None
        ))
    ])

    model.fit(trainX, trainY)

    # Predict probabilities
    predProba = model.predict_proba(testX)

    # Predict class labels (using default threshold of 0.5)
    predClass = model.predict(testX)

    # Evaluate accuracy
    accuracy[i, 0] = accuracy_score(testY, predClass)
