import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from datetime import datetime

def capturar_imagen():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    # reading the input using the camera 
    result, image = cam.read() 
    # If image will detected without any error, 
    if result: 
        # saving image in local storage 
        cv2.imwrite("./imagenes_prueba_modelo/imagen_prueba_1.png", image) 
        return image
    # If captured image is corrupted, moving to else part 
    else: 
        print("No image detected. Please! try again") 

def extraer_coordenadas(imagen)->list|None:
    # Inicializar MediaPipe para detección de manos
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    image_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    # Inicializar la detección de manos
    coordenadas_imagen = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)
        # Si se detectan manos, dibujar los puntos clave en la imagen
        if results.multi_hand_landmarks:
            print('se detectó la mano')
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(imagen, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    coordenadas_imagen.extend([x,y])
                print('se detectaron los 21 puntos de la mano')
    cv2.imshow('Lectura de landmarks',imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coordenadas_imagen
    
def etiquetar_coordenadas(coordenadas_landmarks):
    # El reshape obedece a que el modelo predice sobre un array de numpy de dos dimensiones
    coordenadas_landmarks = np.array(coordenadas_landmarks).reshape((-1,42))
    # Cargar el modelo desde el archivo .h5
    rps_nn = tf.keras.models.load_model('rps_nn.h5')
    # Mostrar un resumen del modelo para verificar que se ha cargado correctamente
    print('Resumen del modelo utilizado para la predicción')
    rps_nn.summary()
    probabilidades = rps_nn.predict(coordenadas_landmarks)[0]
    cod_etiqueta = np.argmax(probabilidades)
    prob_etiqueta = probabilidades[cod_etiqueta]
    return cod_etiqueta,prob_etiqueta

def decodificar_etiqueta(etiqueta:str)->int:
    cod_etiquetas = {
        0 : 'tijera',
        1 : 'piedra',
        2 : 'papel',
    }
    return cod_etiquetas[etiqueta]    

def mostrar_resultados(imagen,etiqueta_predicha:str,prob):
    marca_tiempo = str(datetime.now()).split('.')[0]
    marca_tiempo = marca_tiempo.replace('-','').replace(' ','').replace(':','')
    confianza = int(prob * 100)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    plt.imshow(imagen)
    plt.title(f'Predicción: {etiqueta_predicha.capitalize()}\nConfianza: {confianza}%')
    plt.savefig(f'./imagenes_prueba_modelo/prueba-{marca_tiempo}.jpg')
    plt.show()

img = capturar_imagen()
coordenadas_landmarks = extraer_coordenadas(img)
cod_etiqueta_predicha,prob_etiqueta_predicha = etiquetar_coordenadas(coordenadas_landmarks)
etiqueta_predicha = decodificar_etiqueta(cod_etiqueta_predicha)
mostrar_resultados(imagen=img,etiqueta_predicha=etiqueta_predicha,prob=prob_etiqueta_predicha)
