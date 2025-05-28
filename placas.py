import cv2
import pytesseract
import re
from datetime import datetime

# Configuración básica para pytesseract (requiere instalación)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajustar según tu instalación

def procesar_placa(imagen_path):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        return "Error: No se pudo cargar la imagen"
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtros para mejorar el texto
    filtrado = cv2.bilateralFilter(gris, 11, 17, 17)
    bordes = cv2.Canny(filtrado, 30, 200)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]
    
    placa_contorno = None
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.018 * perimetro, True)
        if len(approx) == 4:  # Buscar contornos con 4 vértices (rectángulo)
            placa_contorno = approx
            break
    
    if placa_contorno is None:
        return "No se detectó ninguna placa en la imagen"
    
    # Extraer la región de la placa
    mascara = np.zeros(gris.shape, np.uint8)
    nueva_imagen = cv2.drawContours(mascara, [placa_contorno], 0, 255, -1)
    nueva_imagen = cv2.bitwise_and(imagen, imagen, mask=mascara)
    
    # Recortar la región de interés
    (x, y) = np.where(mascara == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    placa_recortada = gris[topx:bottomx+1, topy:bottomy+1]
    
    # Usar OCR para leer el texto
    texto = pytesseract.image_to_string(placa_recortada, config='--psm 11')
    texto_limpio = re.sub(r'[^a-zA-Z0-9]', '', texto.upper())
    
    # Analizar la placa
    info = analizar_placa(texto_limpio)
    
    return {
        "placa": texto_limpio,
        "info": info,
        "fecha_consulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def analizar_placa(texto_placa):
    # Esta función analiza la placa y extrae información relevante
    # (Ejemplo para formato mexicano - ajustar según país)
    
    info = {
        "pais": "Desconocido",
        "estado": "Desconocido",
        "tipo_vehiculo": "Desconocido",
        "año": "Desconocido",
        "valida": False
    }
    
    if not texto_placa or len(texto_placa) < 4:
        return info
    
    # Ejemplo para placas mexicanas
    if re.match(r'^[A-Z]{3}\d{3}$', texto_placa):  # Formato ABC-123
        info["pais"] = "México"
        info["valida"] = True
        # El primer caracter puede indicar estado (ejemplo simplificado)
        estados = {
            'A': 'Aguascalientes', 'B': 'Baja California', 
            'C': 'Baja California Sur', 'D': 'Campeche'
            # ... completar con otros estados
        }
        info["estado"] = estados.get(texto_placa[0], "Desconocido")
    
    # Ejemplo para placas de EE.UU. (formato simplificado)
    elif re.match(r'^[A-Z]{2,3}\d{3,4}$', texto_placa):
        info["pais"] = "Estados Unidos"
        info["valida"] = True
    
    # Ejemplo para placas europeas (formato simplificado)
    elif re.match(r'^[A-Z]{2}\d{5}$', texto_placa):
        info["pais"] = "Europa (posible)"
        info["valida"] = True
    
    return info

# Ejemplo de uso
resultado = procesar_placa(r"C:\Users\Uggly Boy\Downloads\Placas\placa1.jpg",0) 
print("Placa detectada:", resultado["placa"])
print("Información:", resultado["info"])