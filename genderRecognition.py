from grabacion import grabar
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import subprocess
import pandas as pd

def procesar_audios(file):
    # Genera voice.csv
    print(file)
    subprocess.call(['RScript','script.r',file+".wav"])
    #proc = subprocess.Popen(['RScript','script.r',file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #stdout, stderr = proc.communicate()

name = raw_input("Introduzca su nombre: ")
name_file = name + "_voice" 
#grabar(20, name_file)
procesar_audios(name_file)
X = pd.read_csv(name_file + ".csv")
forest = joblib.load('random_forest_train.pkl')
gender = forest.predict(X)
if (gender == 0) :
	print(name + "es un HOMBRE.")
else:
	print(name + "es una MUJER.")