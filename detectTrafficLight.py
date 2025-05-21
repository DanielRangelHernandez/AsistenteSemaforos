from ultralytics import YOLO
import cv2

#Se coloca el modelo de aprendizaje
model = YOLO('yolov8n.pt')  # Modelo ligero

# 🎥 Procesar video y generar nuevo video con cuadros marcados
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        #Se debe de hacer la particion del area total en 3 para cada color
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == 'traffic light':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Semaforo', (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
     # Mostrar cámara
    cv2.imshow('camara', frame)


    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()