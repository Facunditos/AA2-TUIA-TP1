import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def carga_datos(dataset,labels)->tuple:
    # Cargar los datos desde un archivo .npy
    X = np.load(dataset)
    y = np.load(labels)
    return X,y

def mostrar_balanceo(etiquetas):
    plt.figure()
    sns.countplot(y=etiquetas)
    plt.title('Balanceo del dataset: frecuencia por categoría')
    plt.yticks(ticks=[0,1,2],labels=['tijera','pidra','papel'])
    plt.xlabel('Frecuencia')
    plt.show()

def dividir_datos(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=132)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=132)
    print('Tamaño de train:',X_train.shape[0])
    print('Tamaño de val:',X_val.shape[0])
    print('Tamaño de test:',X_test.shape[0])
    return X_train,y_train,X_val,y_val,X_test,y_test

class NeuralNetwork:
    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.model = None
        self.history = None
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
    def build_model(self,optimizador,fun_perdida,metricas=None):
        n_features = self.X_train.shape[1]
        n_clases = len(np.unique(self.y_train))
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(units=n_clases, activation='softmax')
        ])
        model.compile(optimizer=optimizador, loss=fun_perdida,metrics=metricas)
        print(model.summary())
        self.model = model
    def train(self,tamaño_lote=None,n_iteraciones=100):
        if tamaño_lote is None:
            tamaño_lote = int(len(self.X_train) ** 0.5)
        history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=tamaño_lote, 
            epochs=n_iteraciones, 
            validation_data=(self.X_val, self.y_val)
        )
        self.history = history
    def graficar_desempeño_entrenamiento(self):
        # Plot the training history, accuracy and loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='entrenamiento')
        plt.plot(self.history.history['val_accuracy'], label = 'validación')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='entrenamiento')
        plt.plot(self.history.history['val_loss'], label = 'validación')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 2])
        plt.legend(loc='upper right')
        plt.suptitle('Evolución del accuracy y de la pérdida en entrenamiento y validación en función a las épocas')
        plt.tight_layout()
        plt.show()
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return accuracy
    def predict(self, X_new):
        predictions = self.model.predict(X_new)
        return predictions
    def save_model(self,nombre):
        self.model.save(f'./{nombre}.h5')
        print(f"Modelo guardado!")

def comparar_accuracies(**kwargs):
    x_plot = kwargs.keys()
    y_plot = kwargs.values()
    y_plot = [round(value,4) for value in y_plot]
    plt.title('Accuracy obtenido según la partición del dataset')
    plt.bar(x_plot,y_plot)
    for i in range(len(x_plot)):
        plt.text(i, y_plot[i], y_plot[i])  # Placing text slightly above the bar
    plt.show()

features_path = './rps_dataset.npy'
target_path = './rps_labels.npy'
X,y = carga_datos(dataset=features_path,labels=target_path)
mostrar_balanceo(y)
X_train,y_train,X_val,y_val,X_test,y_test = dividir_datos(X,y)
rps_nn = NeuralNetwork(X_train,y_train,X_val,y_val,X_test,y_test)
rps_nn.build_model(optimizador='adam',fun_perdida='sparse_categorical_crossentropy',metricas=['accuracy'])
rps_nn.train(tamaño_lote=128,n_iteraciones=90)
rps_nn.graficar_desempeño_entrenamiento()
una_imagen = X_test[0,:]
rps_nn.predict(una_imagen)
acc_train = rps_nn.history.history['accuracy'][-1]
acc_val = rps_nn.history.history['val_accuracy'][-1]
acc_test = rps_nn.evaluate()
comparar_accuracies(train = acc_train, val = acc_val, test = acc_test)
rps_nn.save_model(nombre='rps_nn')