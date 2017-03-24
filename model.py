from grabacion import grabar
from sklearn.ensemble import RandomForestClassifier
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


X,Y = leer_archivo("Entrenamiento/voice.csv")
XT, YT = leer_archivo("Pruebas Random/predict2.csv")
clf = RandomForestClassifier(n_jobs=2)
clf.fit(X, Y)
