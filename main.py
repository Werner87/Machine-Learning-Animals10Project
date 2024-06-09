import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import joblib
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


sciezki = ['raw-img/cane', 'raw-img/cavallo', 'raw-img/elefante', 'raw-img/farfalla', 'raw-img/gallina', 'raw-img/gatto', 'raw-img/mucca', 'raw-img/pecora', 'raw-img/ragno', 'raw-img/scoiattolo']

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}

nazwy_klas = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']

train_photos = []
train_labels = []
desired_shape = (32, 32)

def fillArr(sciezka_idx):
    biblioteka = sciezki[sciezka_idx]
    files = os.listdir(biblioteka)
    for filename in tqdm(files, desc=f'Loading images from class {sciezka_idx}'):
        f = os.path.join(biblioteka, filename)
        if os.path.isfile(f):
            try:
                image = Image.open(f)
                image = image.resize(desired_shape)
                photo_array = np.array(image)
                if photo_array.shape == (32, 32, 3):
                    train_photos.append(photo_array.flatten())
                    train_labels.append(sciezka_idx)
                else:
                    print(f'inny{sciezka_idx}: {photo_array.shape}')
            except Exception as e:
                print(f'Błąd podczas wczytywania obrazu {f}: {e}')

for i in range(0, len(sciezki)):
    fillArr(i)

train_photos = np.array(train_photos)
train_photos = train_photos / 255
train_labels = np.array(train_labels)
print("Zakończono ładowanie obrazów...")

X_train, X_test, y_train, y_test = train_test_split(train_photos, train_labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model SVM
try:
    final_model_svm = joblib.load('best_model_svm.pkl')
    print("Model SVM already fitted.")
except FileNotFoundError:
    print("Model SVM not found. Fitting the model...")
    final_model_svm = SVC(C=10, kernel='rbf', gamma='scale', random_state=42)
    final_model_svm.fit(X_train_scaled, y_train)
    joblib.dump(final_model_svm, 'best_model_svm.pkl')

train_pred_svm = final_model_svm.predict(X_train_scaled)
val_pred_svm = final_model_svm.predict(X_test_scaled)

classification_report_svm = classification_report(y_test, val_pred_svm)
print(f'Final Accuracy for Training (SVM): {np.mean(train_pred_svm == y_train)}')
print(f'Final Accuracy for Validation (SVM): {np.mean(val_pred_svm == y_test)}')
print('Final Classification Report (SVM):\n', classification_report_svm)

# Model KNN
try:
    final_model_knn = joblib.load('best_model_knn.pkl')
    print("Model KNN already fitted.")
except FileNotFoundError:
    print("Model KNN not found. Fitting the model...")
    final_model_knn = KNeighborsClassifier(n_neighbors=3)
    final_model_knn.fit(X_train_scaled, y_train)
    joblib.dump(final_model_knn, 'best_model_knn.pkl')

train_pred_knn = final_model_knn.predict(X_train_scaled)
val_pred_knn = final_model_knn.predict(X_test_scaled)

classification_report_knn = classification_report(y_test, val_pred_knn)
print(f'Final Accuracy for Training (KNN): {np.mean(train_pred_knn == y_train)}')
print(f'Final Accuracy for Validation (KNN): {np.mean(val_pred_knn == y_test)}')
print('Final Classification Report (KNN):\n', classification_report_knn)

import gradio as gr

def classify_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize(desired_shape)
    image_array = np.array(image).flatten().reshape(1, -1) / 255
    image_scaled = scaler.transform(image_array)
    prediction_svm = final_model_svm.predict(image_scaled)
    prediction_knn = final_model_knn.predict(image_scaled)
    return nazwy_klas[prediction_svm[0]], nazwy_klas[prediction_knn[0]]

iface = gr.Interface(fn=classify_image, inputs="image", outputs=[gr.Textbox(label="SVC Prediction"), gr.Textbox(label="KNN Prediction")], live=True)
iface.launch()
