import cv2
import numpy as np

def dibujar(mask, color, frame):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 3), 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 3)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX

# Rangos HSV
rojoBajo1 = np.array([0, 100, 20], np.uint8)
rojoAlto1 = np.array([10, 255, 255], np.uint8)
rojoBajo2 = np.array([160, 100, 20], np.uint8)
rojoAlto2 = np.array([179, 255, 255], np.uint8)
amarilloBajo = np.array([20, 100, 20], np.uint8)
amarilloAlto = np.array([35, 255, 255], np.uint8)
verdeBajo = np.array([36, 100, 20], np.uint8)
verdeAlto = np.array([85, 255, 255], np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear máscaras
    maskRojo1 = cv2.inRange(frameHSV, rojoBajo1, rojoAlto1)
    maskRojo2 = cv2.inRange(frameHSV, rojoBajo2, rojoAlto2)
    maskRojo = cv2.add(maskRojo1, maskRojo2)
    maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
    maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)

    # Obtener valores de brillo (canal V)
    v_channel = frameHSV[:, :, 2]

    brillo_rojo = cv2.mean(v_channel, maskRojo)[0]
    brillo_amarillo = cv2.mean(v_channel, maskAmarillo)[0]
    brillo_verde = cv2.mean(v_channel, maskVerde)[0]

    # Comparar y seleccionar el color más brillante
    brillos = {
        'rojo': (brillo_rojo, maskRojo, (0, 0, 255)),
        'amarillo': (brillo_amarillo, maskAmarillo, (0, 255, 255)),
        'verde': (brillo_verde, maskVerde, (0, 255, 0))
    }

    color_max = max(brillos.items(), key=lambda x: x[1][0])

    # Dibujar solo el color con mayor brillo
    dibujar(color_max[1][1], color_max[1][2], frame)

    # Mostrar cámara
    cv2.imshow('camara', frame)

    # Presiona 's' para salir
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
