from grabacion import grabar
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import subprocess
import pandas as pd
import os

##grabar(5,"hola")
def leer_archivo(nombre):
    df = pd.read_csv(nombre)
    Y = df["label"]
    Yfact = pd.factorize(Y)[0]
    X = df.ix[:,:-1]
    return X, Yfact

def procesar_audios():
    # Genera voice.csv
    proc = subprocess.Popen(['RScript','script.r'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()


X,y = leer_archivo("Entrenamiento/voice.csv")
#print(len(X))
#kf = KFold(n_splits = 5)
#print(kf)
#print(kf.get_n_splits(X))
n_splits = 5
'''
forest2 = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=None, 
								min_samples_split=2, min_samples_leaf=1, 
								min_weight_fraction_leaf=0.0, max_features='auto', 
								max_leaf_nodes=None, min_impurity_split=1e-07, 
								bootstrap=True, oob_score=False, n_jobs=2, 
								random_state=None, verbose=0, warm_start=False, 
								class_weight=None)
score2 = cross_val_score(forest2, X, y, cv=n_splits, n_jobs=2)
print(score2),
print(np.mean(score2))
#forest2.fit(X,y)
#print(forest2.feature_importances_)

tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
							  min_samples_split=2, min_samples_leaf=1, 
							  min_weight_fraction_leaf=0.0, max_features=None, 
							  random_state=None, max_leaf_nodes=None, 
							  min_impurity_split=1e-07, class_weight=None, presort=False)
score = cross_val_score(tree, X, y, cv=n_splits, n_jobs=2)
print(score),
print(np.mean(score))
#tree.fit(X,y)
#print(tree.feature_importances_)
tree2 = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=None, 
							  min_samples_split=2, min_samples_leaf=1, 
							  min_weight_fraction_leaf=0.0, max_features=None, 
							  random_state=0, max_leaf_nodes=None, 
							  min_impurity_split=1e-07, class_weight=None, presort=False)
score2 = cross_val_score(tree, X, y, cv=n_splits, n_jobs=2)
print(score2),
print(np.mean(score2))
#tree2.fit(X,y)
#print(tree2.feature_importances_)

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='logistic', solver='adam', 
					alpha=0.0001, batch_size='auto', learning_rate='constant', 
					learning_rate_init=0.01, power_t=0.5, max_iter=400, shuffle=True, 
					random_state=None, tol=0.0001, verbose=False, warm_start=False, 
					momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
					validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
score = cross_val_score(mlp, X, y, cv=n_splits, n_jobs=2)
print(score),
print(np.mean(score))
'''
'''
for train_index, test_index in kf.split(X):
	print("TRAIN:", train_index, "TEST:", test_index)
	clf.fit(X[train_index], y[train_index])
	#print(clf.score(X[test], y[test]))
'''
forest = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=None, 
								min_samples_split=2, min_samples_leaf=1, 
								min_weight_fraction_leaf=0.0, max_features='auto', 
								max_leaf_nodes=None, min_impurity_split=1e-07, 
								bootstrap=True, oob_score=False, n_jobs=2, 
								random_state=None, verbose=0, warm_start=False, 
								class_weight=None)
print("Random Forest con todos los atributos:")
score = cross_val_score(forest, X, y, cv=n_splits, n_jobs=2)
print ("Precision por conjunto: "),
print(score)
print ("Precision media: "),
print(np.mean(score))
print("")
forest.fit(X,y)
importantFeatures = list(forest.feature_importances_)
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='logistic', solver='adam', 
					alpha=0.0001, batch_size='auto', learning_rate='constant', 
					learning_rate_init=0.01, power_t=0.5, max_iter=400, shuffle=True, 
					random_state=None, tol=0.0001, verbose=False, warm_start=False, 
					momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
					validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
indices = []
for i in range(0, X.shape[1]):
	if (importantFeatures[i] < 0.01) :
		indices.append(i)
X = X.drop(X.columns[indices],axis=1)
for i in sorted(indices, reverse=True):
    del importantFeatures[i]
print("MLP con " + str(len(importantFeatures)) + " atributos:")
print(X.columns)
score = cross_val_score(mlp, X, y, cv=n_splits, n_jobs=2)
print ("Precision por conjunto: "),
print(score)
print ("Precision media: "),
print(np.mean(score))
print("")
while (len(importantFeatures) != 1): 
	minImportance = 10000000000
	minIndex = -1
	for i in range(0,len(importantFeatures)):
		if (importantFeatures[i] < minImportance):
			minImportance = importantFeatures[i]
			minIndex = i
	X = X.drop(X.columns[minIndex],axis=1)
	del importantFeatures[minIndex]
	print("MLP con " + str(len(importantFeatures)) + " atributos:")
	print(X.columns)
	#print(importantFeatures)
	score = cross_val_score(mlp, X, y, cv=n_splits, n_jobs=2)
	print ("Precision por conjunto: "),
	print(score)
	print ("Precision media: "),
	print(np.mean(score))
	print("")