import pandas as pd, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix

def main():
    arqCSV = pd.read_csv('datasets\iris.csv')
    arqCSV = np.array(arqCSV)
    dados = arqCSV[:,:-1].astype(np.float64)
    classes = arqCSV[:,-1]

    nb = GaussianNB()
    dTree = DecisionTreeClassifier(criterion='entropy')
    knn = KNeighborsClassifier(n_neighbors=5)
    svmC = SVC(kernel='poly')

    mets = ['precision_macro','recall_macro','f1_macro','accuracy']
    scoresNaive = cross_validate(nb,dados,classes,cv=5,scoring=mets)
    scoresDtree = cross_validate(dTree,dados,classes,cv=5,scoring=mets)
    scoresKNN = cross_validate(knn,dados,classes,cv=5,scoring=mets)
    scoresSVM = cross_validate(svmC,dados,classes,cv=5,scoring=mets)
    print("Comparacao Árvore de Decisão, Naive Bayes, KNN e SVM")
    for s in scoresNaive:
        print("%s ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f " 
              % (
                  s,
                  np.average(scoresDtree[s]),
                  np.std(scoresDtree[s]),
                  np.average(scoresNaive[s]),
                  np.std(scoresNaive[s]),
                  np.average(scoresKNN[s]),
                  np.std(scoresKNN[s]),
                  np.average(scoresSVM[s]),
                  np.std(scoresSVM[s])
                )
    )

    predCross = cross_val_predict(nb,dados,classes,cv=5)
    confMatrix = confusion_matrix(classes,predCross)
    print("Matriz de confusão para o Naive Bayes")
    print(confMatrix)

    predCross = cross_val_predict(dTree,dados,classes,cv=5)
    confMatrix = confusion_matrix(classes,predCross)
    print("Matriz de confusão para a Árvore de decisão")
    print(confMatrix)

    predCross = cross_val_predict(knn,dados,classes,cv=5)
    confMatrix = confusion_matrix(classes,predCross)
    print("Matriz de confusão para o KNN")
    print(confMatrix)

    predCross = cross_val_predict(svmC,dados,classes,cv=5)
    confMatrix = confusion_matrix(classes,predCross)
    print("Matriz de confusão para o SVM")
    print(confMatrix)

if __name__ == '__main__':
    main()