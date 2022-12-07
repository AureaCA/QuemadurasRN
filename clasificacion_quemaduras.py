# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:22:07 2022

@author: herna
"""

# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras


# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

#Libreria de imagenes
import cv2 as cv


Q11 = np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G1_1.PNG",0))
Q12 = np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G1_2.PNG",0))
Q21= np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G2_1.PNG",0) )
Q22=np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G2_2.PNG",0) )
Q31=np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G2p_1.PNG",0) )
Q32=np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G2p_2.PNG",0) )
Q41=np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G3_1.PNG",0) )
Q42=np.uint8(cv.imread("C:\\Users\\herna\\OneDrive\\Escritorio\\Quemaduras\\train\\G3_2.PNG",0) )

#avion=cv.resize(avion,(carro.shape[1],carro.shape[1]))
Q11=cv.resize(Q11,(300,300))
Q12=cv.resize(Q12,(300,300))
Q21=cv.resize(Q21,(300,300))
Q22=cv.resize(Q22,(300,300))
Q31=cv.resize(Q31,(300,300))
Q32=cv.resize(Q32,(300,300))
Q41=cv.resize(Q41,(300,300))
Q42=cv.resize(Q42,(300,300))

#etiquetas = np.uint8([0,1,2,3])
etiquetas = np.uint8([0,0,1,1,2,2,3,3])

Img = np.uint8([Q11,Q12,Q21,Q22,Q31,Q32,Q41,Q42])
#Img =([Q11,Q21,Q31,Q41]) 


plt.imshow(Q41)
#Img.shape   #shape del vector
plt.imshow(Img[3])

np.array(Img).shape
etiquetas
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(300, 300)),
    keras.layers.Dense(2000, activation='relu'),
    keras.layers.Dense(1000, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(np.array(Img), np.array(etiquetas), epochs=40)
plt.figure()
plt.imshow(Q11)
plt.colorbar()
plt.grid(False)
plt.show()
test_loss, test_acc = model.evaluate(np.array(Img),  np.array(etiquetas), verbose=2)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model.
with open('d:/ia/quemaduras.tflite', 'wb') as f:
  f.write(tflite_model)
  
print('\nTest accuracy:', test_acc)
predictions = model.predict(np.array(Img))
predictions[0]
np.argmax(predictions[0])