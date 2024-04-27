import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix

def breastCancerEstimateDecisionTree():
    arqCSV = pd.read_csv('datasets/lungCancer.csv')
    dados = np.array(arqCSV)
    caracteristicas = dados[:,3:-1].astype(np.float64)
    classes = dados[:,-1]

    dTree = DecisionTreeClassifier(criterion='entropy')
    mets = ['precision_macro','recall_macro','f1_macro','accuracy']
    scores = cross_validate(dTree,caracteristicas,classes,cv=5,scoring=mets)
    print("Métricas ========================")
    for s in scores:
        print("Média da %s => %f === desvio padrão %f" % (s,np.average(scores[s]),np.std(scores[s])))

    predCross = cross_val_predict(dTree,caracteristicas,classes,cv=10)
    confMatrix = confusion_matrix(classes,predCross)
    print(confMatrix)


def main():
    arqCSV = pd.read_csv('datasets/iris.csv')
    dados = np.array(arqCSV)
    caracteristicas = dados[:,:-1].astype(np.float64)
    classes = dados[:,-1]
    dTree = DecisionTreeClassifier(criterion='entropy')

    mets = ['precision_macro','recall_macro','f1_macro','accuracy']

    scores = cross_validate(dTree,caracteristicas,classes,cv=10,scoring=mets)
    print("Métricas ========================")
    for s in scores:
        print("Média da %s => %f === desvio padrão %f" % (s,np.average(scores[s]),np.std(scores[s])))

    predCross = cross_val_predict(dTree,caracteristicas,classes,cv=10)
    confMatrix = confusion_matrix(classes,predCross)
    print(confMatrix)
    #fTreino, fTeste, clasTreino, clasTeste = train_test_split(caracteristicas,classes)

    
    '''
    dTree.fit(fTreino,clasTreino)

    yPred = dTree.predict(fTeste)
    desempenho = [0,0]
    for i in range(len(yPred)):
        desempenho[int(yPred[i] == clasTeste[i])] +=1

    print("acertos: %d === erros %d" % (desempenho[1],desempenho[0]))
    '''
if __name__ == '__main__':
    breastCancerEstimateDecisionTree()