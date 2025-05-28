import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import pytesseract
import mysql.connector
import numpy as np

# Configura la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Conectar a la base de datos
def conectar_base():
    try:
        # Cambia aquí tu contraseña si es diferente
        conn = mysql.connector.connect(
            host="root@localhost",
            port=3306,
            user="root@localhost",
            password="Y0ngShaw3nFoc0", 
            database="placas"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error al conectar a la base de datos: {err}")
        return None

# Guardar la placa en la base de datos
def guardar_placa(placa):
    try:
        conn = conectar_base()
        if conn is None:
            return False
        cursor = conn.cursor()
        sql = "INSERT INTO matrículas (texto) VALUES (%s)"
        cursor.execute(sql, (placa,))
        conn.commit()
        conn.close()
        return True
    except mysql.connector.Error as e:
        print(f"Error guardando en base de datos: {e}")
        return False

# Procesar la imagen para detectar la matrícula
def procesar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        return None

    # Redimensionar para trabajar más rápido
    imagen = cv2.resize(imagen, (800, 600))

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro para quitar ruido
    gris = cv2.bilateralFilter(gris, 11, 17, 17)

    # Detectar bordes
    bordes = cv2.Canny(gris, 30, 200)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos por área (de mayor a menor)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]
    placa_contorno = None

    for contorno in contornos:
        # Aproximar el contorno
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.018 * perimetro, True)

        # Si el contorno tiene 4 lados, podría ser una matrícula
        if len(aprox) == 4:
            placa_contorno = aprox
            break

    if placa_contorno is None:
        print("No se encontró la matrícula.")
        return None

    # Crear una máscara y extraer solo la matrícula
    mascara = np.zeros(gris.shape, np.uint8)
    cv2.drawContours(mascara, [placa_contorno], 0, 255, -1)
    x, y = np.where(mascara == 255)
    if x.size == 0 or y.size == 0:
        print("No se pudo extraer región.")
        return None
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    placa = gris[topx:bottomx+1, topy:bottomy+1]

    # OCR para extraer el texto
    texto = pytesseract.image_to_string(placa, config='--psm 8')

    return texto.strip()

# Función para cargar la imagen desde el menú
def cargar_imagen():
    ruta_imagen = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not ruta_imagen:
        return

    texto_placa = procesar_imagen(ruta_imagen)

    if texto_placa:
        if guardar_placa(texto_placa):
            messagebox.showinfo("Éxito", f"Matrícula detectada: {texto_placa}\nGuardada exitosamente.")
        else:
            messagebox.showerror("Error", "No se pudo guardar en la base de datos.")
    else:
        messagebox.showwarning("Advertencia", "No se pudo detectar ninguna matrícula.")

# Crear la ventana principal
def crear_ventana():
    ventana = tk.Tk()
    ventana.title("Detector de Matrículas")
    ventana.geometry("400x200")

    boton_cargar = tk.Button(ventana, text="Cargar Imagen", command=cargar_imagen, font=("Arial", 14))
    boton_cargar.pack(expand=True)

    ventana.mainloop()

# Ejecutar el programa
if __name__ == "__main__":
    crear_ventana()
