import cv2
import pytesseract
import numpy as np
import mysql.connector

# Configura aquí la ruta de Tesseract si es necesario (en Windows suele ser así)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Función para conectar a la base de datos
def conectar_base():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="tu_contraseña",
            database="tu_base_de_datos",
            unix_socket=None
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error de conexión a la base de datos: {err}")
        return None

# Función para guardar la matrícula en la base de datos
def guardar_placa(placa_texto):
    conn = conectar_base()
    if conn:
        cursor = conn.cursor()
        try:
            sql = "INSERT INTO placas (texto) VALUES (%s)"
            cursor.execute(sql, (placa_texto,))
            conn.commit()
            print("Placa guardada correctamente.")
        except mysql.connector.Error as err:
            print(f"Error al guardar la placa: {err}")
        finally:
            cursor.close()
            conn.close()
    else:
        print("No se pudo guardar porque no hay conexión a la base de datos.")

# Función para detectar y extraer la matrícula
def detectar_matricula(imagen_path):
    # 1. Leer la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        return None

    # 1:08 Transformar a escala de grises y binarizar
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3:40 Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Variables para almacenar el posible rectángulo de matrícula
    placa_contorno = None
    max_area = 0

    # 5:06 Determinar área de los contornos y discriminar
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 4500:  # Puedes ajustar este valor según el tamaño de tu imagen
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = w / float(h)

            # 6:46 Buscar un aspect ratio aproximado de matrícula
            if 2 < aspect_ratio < 6:  # Matrículas son más anchas que altas
                if area > max_area:
                    max_area = area
                    placa_contorno = (x, y, w, h)

    # Si se encontró un contorno candidato
    if placa_contorno:
        x, y, w, h = placa_contorno
        placa = imagen[y:y+h, x:x+w]

        # Opcional: mostrar la placa recortada
        # cv2.imshow('Placa', placa)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return placa
    else:
        print("No se encontró una matrícula.")
        return None

# Función principal para todo el proceso
def procesar_imagen(imagen_path):
    placa_imagen = detectar_matricula(imagen_path)
    if placa_imagen is not None:
        # Aplicar OCR
        texto_placa = pytesseract.image_to_string(placa_imagen, config='--psm 7')  # psm 7 asume una sola línea
        texto_placa = texto_placa.strip()
        print(f"Texto detectado: {texto_placa}")

        if texto_placa:
            guardar_placa(texto_placa)
        else:
            print("No se pudo reconocer texto en la matrícula.")
    else:
        print("No se pudo procesar la imagen.")

# -------------------------
# Ejecución del programa:
if __name__ == "__main__":
    ruta_imagen =r"C:\Users\Uggly Boy\Downloads\Placas\placa1.jpg"
    procesar_imagen(ruta_imagen)
