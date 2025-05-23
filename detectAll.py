import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import pygame
import os
from collections import deque
from time import time

# Inicializar Pygame
pygame.mixer.init()

# Cola de reproducción de audios
cola_sonidos = deque()
reproduciendo = False

# Diccionario de mensajes basado en etiquetas del modelo
mensajes = {
    'bus_stop': 'Parada de bus.',
    'do_not_enter': 'No entrar.',
    'do_not_stop': 'No detenerse.',
    'do_not_turn_l': 'Vuelta prohibida a la izquierda.',
    'do_not_turn_r': 'Vuelta prohibida a la derecha.',
    'do_not_u_turn': 'Vuelta en u prohibida.',
    'green_light': 'Semáforo verde. Avance.',
    'no_parking': 'Prohibido estacionarse.',
    'parking': 'Lugar de estacionamiento.',
    'ped_crossing': 'Cruce de peatones.',
    'ped_zebra_cross': 'Cruce de zebra.',
    'railway_crossing': 'Cruce de trenes.',
    'red_light': 'Semáforo rojo. Alto total.',
    'stop': 'Alto.',
    't_intersection_l': 'Intersección en T.',
    'traffic_light': 'Semáforo.',
    'u_turn': 'Vuelta en U.',
    'warning': 'Precaución. Reduzca velocidad.',
    'yellow_light': 'Semáforo amarillo. Precaución.'
}

# Crear audios si no existen
for etiqueta, mensaje in mensajes.items():
    archivo_audio = f'audios/{etiqueta.lower()}.mp3'
    if not os.path.exists(archivo_audio):
        tts = gTTS(text=mensaje, lang='es')
        tts.save(archivo_audio)

# Función para agregar audio a la cola
def hablar(etiqueta):
    cola_sonidos.append(f'audios/{etiqueta}.mp3')

# Reproduce el siguiente audio en la cola si no se está reproduciendo otro
def manejar_cola():
    global reproduciendo
    if not reproduciendo and cola_sonidos:
        siguiente_audio = cola_sonidos.popleft()
        pygame.mixer.music.load(siguiente_audio)
        pygame.mixer.music.play()
        reproduciendo = True

# Verifica si terminó la reproducción actual
def actualizar_estado():
    global reproduciendo
    if not pygame.mixer.music.get_busy() and reproduciendo:
        reproduciendo = False

# Control de etiquetas ya reproducidas recientemente
ultimas_etiquetas_reproducidas = {}  # {etiqueta: timestamp}
TIEMPO_ESPERA = 5  # segundos antes de volver a decir la misma etiqueta

# Cargar modelo YOLO
model = YOLO('modelo1.pt')

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    etiquetas_detectadas = set()

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        etiquetas_detectadas.add(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, f'Label: {label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Lógica para evitar repeticiones
    momento_actual = time()
    etiquetas_a_reproducir = []

    for etiqueta in etiquetas_detectadas:
        tiempo_ultimo = ultimas_etiquetas_reproducidas.get(etiqueta, 0)
        if momento_actual - tiempo_ultimo > TIEMPO_ESPERA:
            etiquetas_a_reproducir.append(etiqueta)
            ultimas_etiquetas_reproducidas[etiqueta] = momento_actual

    # Agregar a cola
    if len(etiquetas_a_reproducir) == 1:
        hablar(etiquetas_a_reproducir[0])
    elif len(etiquetas_a_reproducir) > 1:
        for etiqueta in etiquetas_a_reproducir:
            hablar(etiqueta)

    manejar_cola()
    actualizar_estado()

    # Mostrar cámara
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
