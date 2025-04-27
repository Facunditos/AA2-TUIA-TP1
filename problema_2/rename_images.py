import os
import shutil
from datetime import datetime
from time import sleep
import re

dir_origen = './imagenes/fotos/Imagenes_mano_tijera'
dir_destino = './imagenes/tijera'

def mover_imagenes(origen,destino):
    for n_img,imagen_ruta in enumerate(os.listdir(origen),1):
        imagen_ruta_origen = origen + '/' + imagen_ruta
        date_time = datetime.now()
        microsegundo = date_time.strftime("%f")
        imagen_ruta_destino = f'{destino}/tijera_{microsegundo}.jpg'
        os.rename(src=imagen_ruta_origen,dst=imagen_ruta_destino)
        # Se pausa la ejecución para que en la sig iteración cambien los microsegundos
        sleep(0.000000000000001)
   

def cambiar_nombre_imagenes(origen):
    figura = dir_origen.split('/')[-1]
    n_imagen = 1
    for n_img,imagen_ruta in enumerate(os.listdir(dir_origen),1):
        imagen_ruta_origen = f'{origen}/{imagen_ruta}'
        imagen_ruta_destino = re.sub(f'_.+',f'_{n_imagen}.jpg',imagen_ruta_origen)
        os.rename(src=imagen_ruta_origen,dst=imagen_ruta_destino)
        n_imagen += 1
        
        

dir_origen = './imagenes/piedra'
cambiar_nombre_imagenes(dir_origen)        

      
    