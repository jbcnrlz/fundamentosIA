import pandas as pd, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix

def main():
    arqCSV = pd.read_csv('datasets\iris.csv')
    arqCSV = np.array(arqCSV)
    dadosTreinoTeste = arqCSV[:,:-1].astype(np.float64)
    classes = arqCSV[:,-1]

    knn = KNeighborsClassifier(n_neighbors=3)
    mets = ['precision_macro','recall_macro','f1_macro','accuracy']
    scores = cross_validate(knn,dadosTreinoTeste,classes,cv=5,scoring=mets)
    for s in scores:
        print("%s ==> Média -> %f | Desvio padrão -> %f" % (
            s,np.average(scores[s]),np.std(scores[s])))
        
    pred = cross_val_predict(knn,dadosTreinoTeste,classes,cv=5)
    confMatrix = confusion_matrix(classes,pred)
    print(confMatrix)

if __name__ == "__main__":
    main()