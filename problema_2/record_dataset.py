import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import warnings
import pandas as pd

def generar_imagen_rgb(img_path):
    image = cv2.imread(img_path)
    # Convertir BGR a RGB (MediaPipe trabaja con imágenes en RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def extraer_coordenadas(imagen)->list|None:
    # Inicializar MediaPipe para detección de manos
    mp_hands = mp.solutions.hands
    # Inicializar la detección de manos
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(imagen)
        # Si se detectan manos, dibujar los puntos clave en la imagen
        if results.multi_hand_landmarks:
            coordenadas_imagen = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    coordenadas_imagen.extend([x,y])
            return coordenadas_imagen
        else:
            return None
    
def codificar_etiqueta(etiqueta:str)->int:
    cod_etiquetas = {
        'tijera' : 0,
        'piedra' : 1,
        'papel' : 2,
    }
    return cod_etiquetas[etiqueta]

def extraer_info_imagenes():
    imagenes_ruta = './imagenes'
    info_imagenes = []
    img_manos_difusas = []
    for figura in os.listdir(imagenes_ruta):
        imagenes_figura = f'{imagenes_ruta}/{figura}'
        for archivo_nombre in os.listdir(imagenes_figura):
            imagen_ruta = f'{imagenes_figura}/{archivo_nombre}'
            image_rgb = generar_imagen_rgb(imagen_ruta)
            coordenadas = extraer_coordenadas(image_rgb)
            if coordenadas is None:
                img_manos_difusas.append(archivo_nombre)  
                continue
            etiqueta = codificar_etiqueta(figura)
            info_imagenes.append({
                'coordenadas':coordenadas,
                'label':etiqueta
            })
    # Guardo un CSV la ruta a las imágenes cuyos landmark no se pudieron procesar 
    pd.Series(img_manos_difusas).to_csv('./img_manos_difusas.csv',index=False,header=False)
    return info_imagenes

def crear_dataset(informacion):
    # Creo el dataset 
    X = [info_imagen['coordenadas'] for info_imagen in informacion]
    X = np.array(X)
    y = [info_imagen['label'] for info_imagen in informacion]
    y = np.array(y)
    print("Dataset creado")
    return X,y

def guardar_dataset(dataset,etiquetas):
    # Guardar el array en un archivo .npy
    np.save('rps_dataset.npy', X)
    np.save('rps_labels.npy', y)
    print("Datos guardados!")


info_imagenes = extraer_info_imagenes()
X,y = crear_dataset(info_imagenes)
guardar_dataset(X,y)





