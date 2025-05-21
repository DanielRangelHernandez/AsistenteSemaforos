import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import pygame
import os

pygame.mixer.init()

# Diccionario de mensajes por color
mensajes = {
    'Rojo': 'Rojo. Alto total.',
    'Amarillo': 'Amarillo. Precaución con frenado.',
    'Verde': 'Verde. Avance.'
}

# Pre-generar audios si no existen
for color, mensaje in mensajes.items():
    archivo_audio = f'{color.lower()}.mp3'
    if not os.path.exists(archivo_audio):
        tts = gTTS(text=mensaje, lang='es')
        tts.save(archivo_audio)

def hablar_color(color):
    archivo_audio = f'{color.lower()}.mp3'
    pygame.mixer.music.load(archivo_audio)
    pygame.mixer.music.play()

def detectar_color_mas_brillante(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    mask_rojo1 = cv2.inRange(hsv, np.array([0, 100, 20], np.uint8), np.array([10, 255, 255], np.uint8))
    mask_rojo2 = cv2.inRange(hsv, np.array([160, 100, 20], np.uint8), np.array([179, 255, 255], np.uint8))
    mask_rojo = cv2.add(mask_rojo1, mask_rojo2)
    mask_amarillo = cv2.inRange(hsv, np.array([20, 100, 20], np.uint8), np.array([35, 255, 255], np.uint8))
    mask_verde = cv2.inRange(hsv, np.array([36, 100, 20], np.uint8), np.array([85, 255, 255], np.uint8))

    brillo_rojo = cv2.mean(v_channel, mask_rojo)[0]
    brillo_amarillo = cv2.mean(v_channel, mask_amarillo)[0]
    brillo_verde = cv2.mean(v_channel, mask_verde)[0]

    brillos = {
        'Rojo': (brillo_rojo, mask_rojo, (0, 0, 255)),
        'Amarillo': (brillo_amarillo, mask_amarillo, (0, 255, 255)),
        'Verde': (brillo_verde, mask_verde, (0, 255, 0))
    }

    #print('Verde ',brillo_verde,' Amarillo ',brillo_amarillo,' Rojo ',brillo_rojo)
    
    return max(brillos.items(), key=lambda x: x[1][0])

def dibujar(mask, color_bgr, roi):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                contorno = cv2.convexHull(c)
                cv2.circle(roi, (x, y), 5, (0, 255, 0), -1)
                cv2.drawContours(roi, [contorno], -1, color_bgr, 2)

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

ultimo_color = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == 'traffic light':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            color_max = detectar_color_mas_brillante(roi)
            nombre_color = color_max[0]
            mascara = color_max[1][1]
            color_bgr = color_max[1][2]

            dibujar(mascara, color_bgr, roi)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f'Semaforo: {nombre_color}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

            if nombre_color != ultimo_color:
                hablar_color(nombre_color)
                ultimo_color = nombre_color

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
