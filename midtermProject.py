import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Training Data 
trainData= pd.read_csv('1990songs_trained_dataset.csv')
print("Shape of the data: ", trainData.shape)
print("Dtypes & Null Values: \n", trainData.info())
print("Class Balance: \n", trainData['Popularity'].value_counts(normalize=True))

# Load Test Data
testData= pd.read_csv('1990songs_test_dataset.csv')
print("\nShape of the data: ", testData.shape)
print("Dtypes & Null Values: \n", testData.info())

# Exploratory Data Analysis (EDA)
numeric_cols = ['Tempo', 'Loudness', 'Danceability', 'Energy', 'Rank']
fig, axes= plt.subplots(1, len(numeric_cols), figsize=(20,10))
for i, col in zip(axes, numeric_cols):
    trainData.groupby('Popularity')[col].plot(kind='kde', ax=i, legend=True, title=col)
plt.tight_layout()
#plt.show()

sns.heatmap(trainData[numeric_cols + ['Popularity']].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
#plt.show()

# Popularity by genre, tempo, loudness, danceability, energy, rank
print("Popularity by Genre: \n", trainData.groupby('Genre')['Popularity'].mean())
trainData.groupby('Genre')['Popularity'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, title='Popularity by Genre')
#plt.show()

print("Popularity by Tempo: \n", trainData.groupby('Tempo')['Popularity'].mean())
trainData.groupby('Tempo')['Popularity'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, title='Popularity by Tempo')
#plt.show()

print("Popularity by Loudness: \n", trainData.groupby('Loudness')['Popularity'].mean())
trainData.groupby('Loudness')['Popularity'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, title='Popularity by Loudness')
#plt.show()

print("Popularity by Danceability: \n", trainData.groupby('Danceability')['Popularity'].mean())
trainData.groupby('Danceability')['Popularity'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, title='Popularity by Danceability')
#plt.show()

print("Popularity by Energy: \n", trainData.groupby('Energy')['Popularity'].mean())
trainData.groupby('Energy')['Popularity'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, title='Popularity by Energy')
#plt.show()

print("Popularity by Rank: \n", trainData.groupby('Rank')['Popularity'].mean())
trainData.groupby('Rank')['Popularity'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, title='Popularity by Rank')
#plt.show()

# Setup Train/Test Split 
important_features = ['Tempo', 'Danceability', 'Energy', 'Rank']
x = trainData[important_features]
y = trainData['Popularity']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train Size: ", len(xTrain), "Test Size: ", len(xTest))
print("Train Approved Rate: ", yTrain.mean(), "Test Approved Rate: ", yTest.mean())

# Classification Models
models={
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    scores= cross_val_score(model, xTrain, yTrain, cv=5, scoring='accuracy')
    print(f'{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')

lr= LogisticRegression(max_iter=1000)
lr.fit(xTrain, yTrain)
lrPred= lr.predict(xTest)

rf= RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(xTrain, yTrain)
rfPred= rf.predict(xTest)

svm= SVC(kernel='rbf', random_state=42)
svm.fit(xTrain, yTrain) 
svmPred= svm.predict(xTest)

# Performance Evaluation
print("\n\nModel Comparison:")
print("Logistic Regression Accuracy: ", accuracy_score(yTest, lrPred))
print("Random Forest Accuracy: ", accuracy_score(yTest, rfPred))
print("Support Vector Machine Accuracy: ", accuracy_score(yTest, svmPred))

print("\n\Classification Report:")
print("Logistic Regression:\n", classification_report(yTest, lrPred))
print("Random Forest:\n", classification_report(yTest, rfPred)) 
print("Support Vector Machine:\n", classification_report(yTest, svmPred))

print("\nConfusion Matrix:")
print("Logistic Regression:\n", confusion_matrix(yTest, lrPred))
print("Random Forest:\n", confusion_matrix(yTest, rfPred))
print("Support Vector Machine:\n", confusion_matrix(yTest, svmPred))

#Predict on Test Data
xTestFinal= testData[important_features]

testPred= lr.predict(xTestFinal)
lrPredFinal= pd.DataFrame({'Title': testData['Title'], 'Artist': testData['Artist'], 'Predicted Popularity': testPred})
lrPredFinal.to_csv('lr_predictions.csv', index=False)
print("\nLogistic Regression Predictions:\n", lrPredFinal)

testPred= rf.predict(xTestFinal)
rfPredFinal= pd.DataFrame({'Title': testData['Title'], 'Artist': testData['Artist'], 'Predicted Popularity': testPred})
rfPredFinal.to_csv('rf_predictions.csv', index=False)
print("\nRandom Forest Predictions:\n", rfPredFinal)

testPred= svm.predict(xTestFinal)
svmPredFinal= pd.DataFrame({'Title': testData['Title'], 'Artist': testData['Artist'], 'Predicted Popularity': testPred})
svmPredFinal.to_csv('svm_predictions.csv', index=False)
print("\nSVM Predictions:\n", svmPredFinal)
