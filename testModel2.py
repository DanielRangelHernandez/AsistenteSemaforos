from ultralytics import YOLO
import cv2

# Cargar modelo
model = YOLO('modelo2.pt')

# Cargar imagen
frame = cv2.imread('signal.jpg')

results = model.predict(source=frame, verbose=False)[0]

if results.boxes is not None and len(results.boxes) > 0:
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Reproduce solo una vez por tipo detectado
                #match label:
                #   case 'red_light':
                #      hablar_color('rojo')
                # case 'yellow_light':
                    #    hablar_color('amarillo')
                    #case 'green_light':
                    #    hablar_color('verde')

        cv2.putText(frame, f'{label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Mostrar la imagen con resultados
cv2.imshow('Resultado', frame)
cv2.waitKey(0)  # Espera hasta que presiones una tecla
cv2.destroyAllWindows()