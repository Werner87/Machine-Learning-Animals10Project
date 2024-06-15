import gradio as gr
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib

svm_model_path = 'best_model_svm.pkl'
knn_model_path = 'best_model_knn.pkl'

nazwy_klas = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']

scaler = StandardScaler()

try:
    final_model_svm = joblib.load(svm_model_path)
    print("Model SVM załadowany pomyślnie.")
except FileNotFoundError:
    print("Model SVM nie znaleziony.")
    final_model_svm = None

try:
    final_model_knn = joblib.load(knn_model_path)
    print("Model KNN załadowany pomyślnie.")
except FileNotFoundError:
    print("Model KNN nie znaleziony.")
    final_model_knn = None

try:
    scaler = joblib.load('scaler.pkl')
    print("Skaler załadowany pomyślnie.")
except FileNotFoundError:
    print("Skaler nie znaleziony.")
    scaler = None

desired_shape = (32, 32)

def classify_image(image):
    if final_model_svm is None or final_model_knn is None:
        return "Modele nie załadowane", "Modele nie załadowane"

    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize(desired_shape)
    image_array = np.array(image).flatten().reshape(1, -1) / 255
    image_scaled = scaler.transform(image_array)
    
    prediction_svm = final_model_svm.predict(image_scaled)
    prediction_knn = final_model_knn.predict(image_scaled)
    
    return nazwy_klas[prediction_svm[0]], nazwy_klas[prediction_knn[0]]

iface = gr.Interface(
    fn=classify_image, 
    inputs="image", 
    outputs=[gr.Textbox(label="Predykcja SVC"), gr.Textbox(label="Predykcja KNN")], 
    live=True
)

iface.launch()