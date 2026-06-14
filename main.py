import cv2
import urllib.request
import numpy as np
import easyocr
import re
import os
from datetime import datetime, timedelta
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("autocheck-esp32-cam-firebase-adminsdk-74itm-a8b6949ea7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


ESP32_URL = "http://192.168.1.15/cam-hi.jpg"
CAMARA_ID = "ESP32_CAM_01"


MINUTOS_DUPLICADO = 5        # Tiempo mínimo entre detecciones de la misma placa
CONFIANZA_MINIMA  = 0.5      # Confianza mínima para aceptar detección de YOLO

# CARGAR MODELOS
# YOLOv8s con pesos preentrenados para detección de placas
print("[INFO] Cargando modelo YOLOv8s...")
model = YOLO("yolov8s.pt")

print("[INFO] Inicializando EasyOCR...")
reader = easyocr.Reader(["es", "en"], gpu=False)

registro_local = {}


def es_placa_valida(texto: str) -> bool:
    """
    Valida si el texto detectado corresponde a un formato
    de placa vehicular mexicana.
    """
    patron = (
        r'^([A-Z]{3}-\d{4})|'          # ABC-1234
        r'^(\d{2}-[A-Z]{3}-\d{2})|'    # 12-ABC-34
        r'^([A-Z]{3}\d{2,3})|'         # ABC12 o ABC123
        r'^(\d{3}-[A-Z]{3})|'          # 123-ABC
        r'^(\d{3}[A-Z]{3})|'           # 123ABC
        r'^(G\d{3}-[A-Z]{3})|'         # G123-ABC
        r'^([A-Z]{3}-\d{3}-[A-Z])$'    # ABC-123-A
    )
    return bool(re.match(patron, texto))


def es_duplicado(placa: str) -> bool:
    """
    Verifica si la placa fue detectada hace menos de MINUTOS_DUPLICADO.
    Retorna True si se debe ignorar, False si se debe registrar.
    """
    if placa not in registro_local:
        return False

    ultima = registro_local[placa]
    diferencia = datetime.now() - ultima
    return diferencia < timedelta(minutes=MINUTOS_DUPLICADO)


def guardar_en_firestore(placa: str, confianza: float):
    """
    Guarda la detección en Firebase Firestore.
    Colección: detecciones
    """
    ahora = datetime.now()

    doc = {
        "placa":      placa,
        "confianza":  round(confianza * 100, 2),
        "fecha_hora": ahora.strftime("%Y-%m-%dT%H:%M:%S"),
        "camara":     CAMARA_ID,
        # imagen_url: se puede agregar cuando se integre Firebase Storage
    }

    
    db.collection("detecciones").add(doc)

    print(f"[FIRESTORE] Guardado: {doc}")


def detectar_y_leer(img: np.ndarray):
    """
    Pipeline principal:
    1. YOLOv8s detecta regiones candidatas a placa
    2. EasyOCR lee el texto de cada región
    3. Se valida el formato
    4. Se controla duplicados
    5. Se guarda en Firestore
    """
    resultados = model(img, conf=CONFIANZA_MINIMA, verbose=False)

    for resultado in resultados:
        for box in resultado.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianza = float(box.conf[0])

            # Recortar región detectada
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Preprocesamiento básico para mejorar OCR
            roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gris = cv2.resize(roi_gris, (300, 100))
            roi_gris = cv2.bilateralFilter(roi_gris, 11, 17, 17)

            # Leer texto con EasyOCR
            ocr_result = reader.readtext(roi_gris, detail=0)
            texto_raw  = " ".join(ocr_result).upper().replace(" ", "")

            if not texto_raw:
                continue

            print(f"[OCR] Texto detectado: {texto_raw} | Confianza YOLO: {confianza:.2f}")

            # Validar formato de placa
            if not es_placa_valida(texto_raw):
                print(f"[SKIP] '{texto_raw}' no es una placa válida")
                continue

            # Controlar duplicados
            if es_duplicado(texto_raw):
                print(f"[SKIP] '{texto_raw}' ya fue registrada recientemente")
                continue

            # Registrar y guardar
            registro_local[texto_raw] = datetime.now()
            guardar_en_firestore(texto_raw, confianza)

            # Dibujar bounding box en la imagen
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, texto_raw,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

    return img


print("[INFO] Iniciando sistema AutoCheck...")
print(f"[INFO] Conectando a ESP32-CAM: {ESP32_URL}")

while True:
    try:
        # Capturar frame del ESP32-CAM
        respuesta   = urllib.request.urlopen(ESP32_URL, timeout=5)
        img_bytes   = np.array(bytearray(respuesta.read()), dtype=np.uint8)
        frame       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            print("[WARN] Frame vacío, reintentando...")
            continue

        # Procesar frame
        frame_procesado = detectar_y_leer(frame)

        # Mostrar resultado
        cv2.imshow("AutoCheck — ESP32-CAM", frame_procesado)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            print("[INFO] Cerrando sistema...")
            break

    except urllib.error.URLError:
        print(f"[ERROR] No se puede conectar al ESP32-CAM en {ESP32_URL}")
        print("[INFO]  Verifica que el ESP32 esté encendido y en la misma red")

    except KeyboardInterrupt:
        print("[INFO] Interrumpido por el usuario")
        break

    except Exception as e:
        print(f"[ERROR] {e}")

cv2.destroyAllWindows()