from ultralytics import YOLO
import cv2
import pygame

pygame.mixer.init()

def hablar_color(color):
    archivo_audio = f'{color.lower()}.mp3'
    pygame.mixer.music.load(archivo_audio)
    pygame.mixer.music.play()

# Cargar modelo
# El modelo debe de quedar mejor manejado dado que solo detecta ciertas señales
model = YOLO('best.pt')  # o 'best.pt' si lo tienes entrenado

# Abrir cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Predicción
    results = model.predict(source=frame, verbose=False)[0]

    # Dibujo de cajas
    
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #Modificar para que solo suene una vez
            match label:
                case 'red_light':
                    hablar_color('rojo')
                case 'yellow_light':
                    hablar_color('amarillo')
                case 'green_light':
                    hablar_color('verde')
                    
            cv2.putText(frame, f'{label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Mostrar frame
    cv2.imshow('camara', frame)

    key = cv2.waitKey(1)
    if key == ord('s') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
