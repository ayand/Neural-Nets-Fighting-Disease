import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing dataset and extracting features and labels
dataset = pd.read_csv('data.csv')
features = dataset.iloc[:, 2:32].values
labels = dataset.iloc[:, 1].values

# Deriving training and testing sets
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3)

# Normalizing the features
normalizer = StandardScaler()
feature_train = normalizer.fit_transform(feature_train)
feature_test = normalizer.transform(feature_test)

# Making the classifier
neuralNet = Sequential()
neuralNet.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_shape=(30,)))
neuralNet.add(Dense(units=20, kernel_initializer='uniform', activation='relu'))
neuralNet.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
neuralNet.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Training the classifier
neuralNet.fit(feature_train, label_train, batch_size=20, epochs=15)

# Testing the classifier and seeing the confusion matrix
predictions = neuralNet.predict(feature_test)
label_test = np.array(label_test)
predictions = (predictions > 0.5)
"""print(str(type(label_test[0])))
print(str(type(predictions[0])))"""
confusionMatrix = confusion_matrix(label_test, predictions)
print("Confusion Matrix:")
print(str(confusionMatrix))

neuralNet.save("breast_cancer_prediction_model.hdf5")
