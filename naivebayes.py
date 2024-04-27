import pandas as pd, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def loadDataset(pathDataset):
    datasetCSV = pd.read_csv(pathDataset)
    data = np.array(datasetCSV)
    return data[:,:-1].astype(np.float64), data[:,-1], datasetCSV.columns

def main():
    data, classes, cols = loadDataset('datasets/iris.csv')
    le = preprocessing.LabelEncoder()
    yvs = le.fit_transform(classes)

    X_train, X_test, y_train, y_test = train_test_split(data, yvs, random_state=0)

    deClass = GaussianNB()
    deClass.fit(X_train,y_train)
    guessed = deClass.predict(X_test)
    print(classification_report(y_test, guessed))
    
if __name__ == '__main__':
    main()