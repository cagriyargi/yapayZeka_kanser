from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = read_csv("C:\\cancer.csv")
# data.info()
# İstatistik / Veriler

# data.Result.value_counts()[0:30].plot(kind = 'bar', color = ['red', 'blue'])
# plt.show()
# data.Result.value_counts()[0:30].plot(kind = 'pie', colors = ['red', 'blue'], explode = [0, 0.1])
# plt.show()
# Tanı Dağılımı

###############################################################################################################

# sns.set_style('whitegrid')
# Kareli Arkaplan
# sns.pairplot(data, hue = 'Sonuc', height = 1, palette = 'rocket_r')
# plt.show()
# Veri Görselleştirme

###############################################################################################################

droppedData = data.drop(columns = ['Ad', 'Soyad'], axis = 1)
droppedData = droppedData.dropna(how = 'any')
# print(droppedData.head())
# Ad, Soyad kolonlarının çıkartılmış hali

###############################################################################################################

Y = droppedData['Sonuc']
X = droppedData.drop(columns=['Sonuc'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# sns.lineplot(x = 'Yas', y = 'Sonuc', hue = 'Sigara', data = droppedData) # Sigara alışkanlığı olan kişilere göre Yas ve Sonuc ilişkisi
# plt.show()

# sns.lineplot(x = 'Yas', y = 'Sonuc', hue = 'Alkol', data = droppedData) # Alkol alışkanlığı olan kişilere göre Yas ve Sonuc ilişkisi
# plt.show()

# sns.lineplot(x = 'Yas', y = 'Sonuc', hue = 'AlanKalitesi', data = droppedData) # AlanKalitesi'nin Yas ve Sonuc ilişkisi
# plt.show()

###############################################################################################################
### KOMŞULUK ALGORİTMASI ######################################################################################
###############################################################################################################

# Modeli tanımladık
komsuModeli = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)

# Modeli eğittik
komsuModeli.fit(X_train, Y_train)

# Sonucu aldık
Y_predictKomsu = komsuModeli.predict(X_test)

# Hata Matrisinin Oluşturulması
komsuModeliMatrix = confusion_matrix(Y_test, Y_predictKomsu)
f, ax = plt.subplots(figsize = (5, 5))

sns.heatmap(komsuModeliMatrix, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
plt.title('KNN Algoritması - Hata Matrisi')
plt.xlabel('Y Tahmin')
plt.ylabel('Y Test')
# plt.show()

komsuSkoru = komsuModeli.score(X_test, Y_test)
print("Komşuluk Algoritması Skoru: ", komsuSkoru)

###############################################################################################################
### LOJİSTİK REGRESYON ########################################################################################
###############################################################################################################

# Modeli tanımladık
lReg = LogisticRegression(C = 10)

# Modeli eğittik
lReg.fit(X_train, Y_train)

# Sonucu aldık
Y_predictLog = lReg.predict(X_test)

# Hata Matrisinin Oluşturulması
lRegMatrix = confusion_matrix(Y_test, Y_predictLog)
f, ax = plt.subplots(figsize = (5, 5))

sns.heatmap(lRegMatrix, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
plt.title('Lojistik Regresyon Algoritması - Hata Matrisi')
plt.xlabel('Y Tahmin')
plt.ylabel('Y Test')
# plt.show()

lRegSkor = lReg.score(X_test, Y_test)
print("Lojistik Regresyon Skoru: ", lRegSkor)

###############################################################################################################
### KARAR AĞACI ###############################################################################################
###############################################################################################################

# Modeli tanımladık
kararAgaci = DecisionTreeClassifier(random_state = 9)

# Modeli eğittik
kararAgaci.fit(X_train, Y_train)

# Sonucu aldık
Y_predictAgac = kararAgaci.predict(X_test)

# Hata Matrisinin Oluşturulması
agacMatrix = confusion_matrix(Y_test, Y_predictAgac)
f, ax = plt.subplots(figsize = (5, 5))

sns.heatmap(agacMatrix, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
plt.title('Karar Ağacı Algoritması - Hata Matrisi')
plt.xlabel('Y Tahmin')
plt.ylabel('Y Test')
# plt.show()

kararAgaciSkor = kararAgaci.score(X_test, Y_test)
print("Karar Ağacı Skoru: ", kararAgaciSkor)

###############################################################################################################
### RASGELE ORMAN ALGORİTMASI #################################################################################
###############################################################################################################

# Modeli tanımladık
rOrman = RandomForestClassifier(n_estimators = 100, random_state = 9, n_jobs = -1)

# Modeli eğittik
rOrman.fit(X_train, Y_train)

# Sonucu aldık
Y_predictOrman = rOrman.predict(X_test)

# Hata Matrisinin Oluşturulması
ormanMatrix = confusion_matrix(Y_test, Y_predictOrman)
f, ax = plt.subplots(figsize = (5, 5))

sns.heatmap(ormanMatrix, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
plt.title('Rasgele Orman Algoritması - Hata Matrisi')
plt.xlabel('Y Tahmin')
plt.ylabel('Y Test')
# plt.show()

ormanSkor = rOrman.score(X_test, Y_test)
print("Rasgele Orman Algoritması Skoru: ", ormanSkor)

###############################################################################################################
### N. BAYES ALGORİTMASI ######################################################################################
###############################################################################################################

# Modeli tanımladık
nBayes = GaussianNB()

# Modeli eğittik
nBayes.fit(X_train, Y_train)

# Sonucu aldık
Y_predictBayes = nBayes.predict(X_test)

# Hata Matrisinin Oluşturulması
bayesMatrix = confusion_matrix(Y_test, Y_predictBayes)
f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(bayesMatrix, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
plt.title('Naive Bayes Algoritması - Hata Matrisi')
plt.xlabel('Y Tahmin')
plt.ylabel('Y Test')
# plt.show()

bayesSkor = nBayes.score(X_test, Y_test)
print("N. Bayes Algoritması Skoru: ", bayesSkor)

###############################################################################################################
### SKOR TABLOSU ##############################################################################################
###############################################################################################################

list_models = ["Komşuluk Algoritması", "Lojistik Regresyon", "Karar Ağacı", "Rasgele Orman Algoritması", "N. Bayes Algoritması"]
list_scores = [komsuSkoru, lRegSkor, kararAgaciSkor, ormanSkor, bayesSkor]

plt.figure(figsize = (12, 4))
plt.bar(list_models, list_scores, width = 0.2, color = ['red', 'blue', 'brown', 'purple', 'orange'])
plt.title('Algoritma - Skor Oranı')
plt.xlabel('Algoritmalar')
plt.ylabel('Skorlar')
plt.show()
