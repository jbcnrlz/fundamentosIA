import pandas as pd, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix

def main():
    irisData = pd.read_csv('datasets\iris.csv')
    irisData = np.array(irisData)
    dados = irisData[:,:-1]
    classes = irisData[:,-1]

    svmClass = SVC(kernel='poly')
    mets = ['precision_macro','recall_macro','f1_macro','accuracy']
    scores = cross_validate(svmClass,dados,classes,cv=5,scoring=mets)
    for s in scores:
        print("%s = Média %f Desvio padrão %f" % (s,np.average(scores[s]),np.std(scores[s])))

    preds = cross_val_predict(svmClass,dados,classes,cv=5)
    cMAtrix = confusion_matrix(classes,preds)
    print(cMAtrix)

if __name__ == '__main__':
    main()