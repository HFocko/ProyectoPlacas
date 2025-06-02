import cv2
import pytesseract
import numpy as np
import os
import re
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
import joblib
from collections import Counter

class RobustPlateRecognizer:
    def __init__(self):
        # Configuración de Tesseract
        # Asegúrate de que esta ruta es correcta para tu instalación de Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Modelos y codificadores
        self.mlp_model = None
        self.label_encoder = None
        self.load_models()
        
        # Patrones de validación para placas mexicanas
        self.plate_patterns = [
            r'^[A-Z]{3}\d{3}$',  # Formato tradicional ABC-123
            r'^[A-Z]{2}\d{4}$',  # Formato nuevo AB-1234
            r'^[A-Z]{3}\d{2}[A-Z]{2}$',  # Placas especiales
            r'^[A-Z]{2}\d{3}[A-Z]{2}$'   # Placas federales
        ]
        
        # Parámetros optimizados
        self.min_plate_aspect_ratio = 1.8
        self.max_plate_aspect_ratio = 5.5
        self.min_plate_area = 1500
        self.max_plate_area = 10000
        
        # Diccionario de estados por código
        self.state_codes = {
            'A': 'Aguascalientes', 'B': 'Baja California', 
            'C': 'Baja California Sur', 'D': 'Campeche',
            'E': 'Coahuila', 'F': 'Colima', 'G': 'Chiapas',
            'H': 'Chihuahua', 'J': 'Durango', 'K': 'Guanajuato',
            'L': 'Guerrero', 'M': 'Estado de México', 
            'N': 'Michoacán', 'O': 'Morelos', 'P': 'Nayarit',
            'Q': 'Nuevo León', 'R': 'Oaxaca', 'S': 'Puebla',
            'T': 'Querétaro', 'U': 'Quintana Roo', 
            'V': 'San Luis Potosí', 'W': 'Sinaloa', 
            'X': 'Sonora', 'Y': 'Tabasco', 'Z': 'Tamaulipas',
            'CDMX': 'Ciudad de México'
        }

        # Configuración de OCR
        self.ocr_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    def load_models(self):
        """Carga modelos ML preentrenados"""
        try:
            # Asegúrate que estas rutas sean correctas para tus modelos
            self.mlp_model = joblib.load('mlp_plate_char_classifier.joblib')
            self.label_encoder = joblib.load('label_encoder.joblib')
            print("Modelos MLP y LabelEncoder cargados exitosamente.")
        except FileNotFoundError as e:
            print(f"Advertencia: {e} - No se encontraron los archivos del modelo. Continuando solo con OCR.")
            self.mlp_model = None
            self.label_encoder = None # Asegurarse de que label_encoder también sea None si no se carga
        except Exception as e:
            print(f"Advertencia: Error al cargar modelos: {str(e)} - Continuando solo con OCR.")
            self.mlp_model = None
            self.label_encoder = None

    def adaptive_preprocessing(self, image):
        """Preprocesamiento adaptativo basado en análisis de imagen"""
        # Convertir a LAB y ecualizar el canal L
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Detectar condiciones de iluminación
        l_mean = np.mean(l)
        if l_mean < 85:  # Imagen oscura
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
        elif l_mean > 170:  # Imagen muy clara
            l_enhanced = cv2.addWeighted(l, 0.7, np.zeros_like(l), 0, 30)
        else:  # Iluminación normal
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
        
        # Reconstruir imagen LAB
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        
        # Binarización adaptativa basada en análisis de histograma
        if np.std(gray) < 25:  # Bajo contraste
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 5)
        else:
            _, binary = cv2.threshold(gray, 0, 255, 
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas adaptativas
        kernel_size = 3 if image.shape[1] > 1000 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return morph

    def detect_plate_candidates(self, image):
        """Detección mejorada de candidatos a placa"""
        processed = self.adaptive_preprocessing(image)
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = w * h
            
            # Filtrado mejorado de regiones
            if (self.min_plate_aspect_ratio < aspect_ratio < self.max_plate_aspect_ratio and 
                self.min_plate_area < area < self.max_plate_area):
                
                # Calcular solidez del contorno
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Calcular extensión
                rect_area = w * h
                extent = float(area) / rect_area
                
                # Puntuación del candidato
                # Ajustamos el factor para el aspecto, favoreciendo más los rangos ideales
                score = solidity * extent * (1.0 - abs(aspect_ratio - 3.5) / 3.5) 
                
                candidates.append({
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'score': score
                })
        
        # Ordenar candidatos por puntuación
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def extract_plate_region(self, image, candidate):
        """Extrae y corrige perspectiva de la placa"""
        x, y, w, h = candidate['bbox']
        # Usamos la imagen original para extraer la región, no la preprocesada
        plate_img_raw = image[y:y+h, x:x+w].copy() 
        
        # Corrección de perspectiva para placas inclinadas
        rect = cv2.minAreaRect(candidate['contour'])
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Cambio de np.int0 a np.intp para compatibilidad
        
        # Ordenar puntos para transformación
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect
            
        ordered = order_points(box)
        (tl, tr, br, bl) = ordered
        
        # Calcular nuevo ancho y alto
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Puntos de destino para la transformación
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Aplicar transformación de perspectiva
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) # Usamos la imagen original aquí también
        
        return warped

    def validate_plate_format(self, text):
        """Valida el formato de la placa y calcula confianza"""
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not text:
            return False, 0.0
        
        best_match_ratio = 0.0
        is_valid_format = False
        
        # Verificar contra todos los patrones
        for pattern in self.plate_patterns:
            match = re.match(pattern, text)
            if match:
                matched_chars = len(match.group())
                current_ratio = matched_chars / len(text)
                if current_ratio > best_match_ratio:
                    best_match_ratio = current_ratio
                is_valid_format = True # Al menos un patrón coincide
        
        # Si el texto no coincide con ningún patrón, la confianza es baja
        if not is_valid_format:
            return False, 0.1 # Confianza muy baja si no coincide con ningún patrón
            
        # Calcular confianza basada en la mejor coincidencia
        confidence = min(1.0, best_match_ratio * 1.2) # Multiplicador para ajustar
        
        # Reglas adicionales para aumentar confianza
        # Considerar la longitud, las placas mexicanas tienen longitudes específicas
        if 6 <= len(text) <= 7: # Longitudes comunes de placas mexicanas
            confidence = min(1.0, confidence + 0.1)
        
        # Verificar código de estado si está presente (primer carácter)
        if text and text[0] in self.state_codes:
            confidence = min(1.0, confidence + 0.15)
        
        return is_valid_format, confidence

    def postprocess_text(self, text):
        """Corrección inteligente de errores comunes en OCR"""
        if not text:
            return ""
        
        # Limpieza básica
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Corrección de caracteres comúnmente confundidos
        char_map = {
            '0': 'O', 'O': '0',
            '1': 'I', 'I': '1',
            '2': 'Z', 'Z': '2',
            '4': 'A', 'A': '4',
            '5': 'S', 'S': '5',
            '6': 'G', 'G': '6',
            '7': 'Z',
            '8': 'B', 'B': '8'
        }
        
        # Aplicar correcciones solo si mejoran la validación
        original_valid, original_conf = self.validate_plate_format(clean_text)
        best_text = clean_text
        best_conf = original_conf
        
        # Generar posibles variantes
        variants = [clean_text]
        for i, char in enumerate(clean_text):
            if char in char_map:
                # Intenta la corrección simple
                variant_simple = clean_text[:i] + char_map[char] + clean_text[i+1:]
                variants.append(variant_simple)
                
                # Para algunos caracteres, intenta la otra dirección (ej. O a 0)
                if char_map.get(char_map[char]) == char: # Si es una conversión bidireccional
                     pass # Ya está cubierta si lo mapeamos en ambos sentidos
                
        # Evaluar todas las variantes
        for variant in list(set(variants)): # Usar set para evitar duplicados
            _, variant_conf = self.validate_plate_format(variant)
            if variant_conf > best_conf:
                best_text = variant
                best_conf = variant_conf
        
        return best_text

    def recognize_with_confidence(self, plate_image):
        """Reconocimiento con estimación de confianza"""
        # OCR tradicional con configuración optimizada
        ocr_text = pytesseract.image_to_string(plate_image, config=self.ocr_config)
        ocr_processed = self.postprocess_text(ocr_text)
        ocr_valid, ocr_conf = self.validate_plate_format(ocr_processed)
        
        # Inicializar variables para MLP para evitar el error 'unbound local variable'
        mlp_text = None
        mlp_conf = 0.0
        
        # Reconocimiento por caracteres si MLP está disponible
        if self.mlp_model and self.label_encoder: # Asegurarse que ambos están cargados
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            # Mejorar binarización para el MLP
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Dilatación para conectar caracteres rotos, si es necesario
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.dilate(binary, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_images = []
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filtro de tamaño de caracteres más preciso
                min_char_height = plate_image.shape[0] * 0.4 # Carácter al menos 40% de la altura de la placa
                max_char_height = plate_image.shape[0] * 0.9 # Carácter no más del 90%
                min_char_width = plate_image.shape[1] * 0.05 # Carácter al menos 5% del ancho de la placa
                max_char_width = plate_image.shape[1] * 0.3 # Carácter no más del 30%

                if min_char_width < w < max_char_width and min_char_height < h < max_char_height: 
                    char_img = gray[y:y+h, x:x+w]
                    char_images.append((x, char_img))
            
            if char_images:
                char_images.sort(key=lambda x: x[0])
                mlp_chars = []
                char_confs = []
                
                for _, char_img in char_images:
                    # Normalizar tamaño para HOG
                    resized = cv2.resize(char_img, (20, 20), interpolation=cv2.INTER_AREA)
                    
                    # Calcular HOG features
                    features = hog(resized, orientations=9, pixels_per_cell=(8,8),
                                   cells_per_block=(2,2), transform_sqrt=True, block_norm='L2-Hys') # block_norm para mayor robustez
                    
                    # Asegurarse de que las features tienen la forma esperada por el modelo
                    features = features.reshape(1, -1)
                    
                    try:
                        probas = self.mlp_model.predict_proba(features)[0]
                        # Asegurarse de que hay al menos 2 clases para top2
                        if len(probas) >= 2:
                            top_indices = np.argsort(probas)[-2:]
                            main_char_idx = top_indices[-1]
                            sec_char_idx = top_indices[-2]
                            
                            main_char = self.label_encoder.inverse_transform([main_char_idx])[0]
                            main_conf = probas[main_char_idx]
                            
                            sec_char = self.label_encoder.inverse_transform([sec_char_idx])[0]
                            sec_conf = probas[sec_char_idx]
                            
                            # Aplicar reglas de confusión común
                            confusions = {
                                '0': 'O', '1': 'I', '2': 'Z', '5': 'S',
                                '6': 'G', '7': 'Z', '8': 'B'
                            }
                            
                            # Si el carácter principal es un número y el segundo mejor es su letra confundible
                            # O viceversa, y las confianzas son cercanas
                            if (main_char in confusions and sec_char == confusions[main_char] and sec_conf > main_conf * 0.7) or \
                               (sec_char in confusions and main_char == confusions[sec_char] and main_conf > sec_conf * 0.7):
                                # Decidir cuál es más probable en el contexto de una placa
                                # Por ahora, vamos con el que tenga mayor confianza
                                if sec_conf > main_conf:
                                    main_char = sec_char
                                    main_conf = sec_conf

                            mlp_chars.append(main_char)
                            char_confs.append(main_conf)
                        else: # Si solo hay una clase predicha o menos de 2
                            main_char_idx = np.argmax(probas)
                            mlp_chars.append(self.label_encoder.inverse_transform([main_char_idx])[0])
                            char_confs.append(probas[main_char_idx])

                    except Exception as e:
                        print(f"Error en la predicción MLP para un carácter: {e}")
                        # Si hay un error en la predicción, asigna un caracter genérico de baja confianza
                        mlp_chars.append('?') 
                        char_confs.append(0.1)
                
                if mlp_chars:
                    mlp_text = ''.join(mlp_chars)
                    mlp_processed = self.postprocess_text(mlp_text)
                    mlp_valid, mlp_conf = self.validate_plate_format(mlp_processed)
                    mlp_conf *= np.mean(char_confs) if char_confs else 0.0 # Confianza promedio de caracteres

        # Combinar resultados
        final_text = None
        final_conf = 0.0
        
        # Priorizar MLP si está disponible y da un resultado válido con buena confianza
        if self.mlp_model and mlp_text and mlp_conf > ocr_conf:
            final_text = mlp_processed
            final_conf = mlp_conf
        else: # Si no hay MLP, o el OCR es mejor, usa OCR
            final_text = ocr_processed
            final_conf = ocr_conf
        
        # Post-procesamiento final basado en confianza, si aún es baja
        if final_text and final_conf < 0.7:
            # Intentar combinaciones si la confianza es baja
            if ocr_processed and mlp_text and ocr_processed != mlp_text:
                combined_candidates = set()
                # Combina caracteres de ambos resultados si son diferentes
                min_len = min(len(ocr_processed), len(mlp_text))
                
                for i in range(min_len):
                    if ocr_processed[i] == mlp_text[i]:
                        combined_candidates.add(ocr_processed[i])
                    else:
                        combined_candidates.add(ocr_processed[i])
                        combined_candidates.add(mlp_text[i])
                
                # Esto es una simplificación, una estrategia más avanzada sería un algoritmo de votación
                # Por ahora, simplemente tomamos el mejor de los dos si el resultado inicial es bajo
                if ocr_conf > mlp_conf and ocr_conf > final_conf:
                    final_text = ocr_processed
                    final_conf = ocr_conf
                elif mlp_text and mlp_conf > final_conf:
                    final_text = mlp_processed
                    final_conf = mlp_conf
        
        # Asegurarse de que la confianza no sea nula si hay texto pero no se validó bien
        if final_text and final_conf == 0.0:
            _, final_conf = self.validate_plate_format(final_text)

        return final_text, final_conf

    def detect_state_from_plate(self, plate_text):
        """Detecta el estado basado en el primer carácter de la placa"""
        if not plate_text:
            return "No detectado", 0.0
        
        # Verificar código de estado en primer carácter (si no es CDMX)
        first_char = plate_text[0]
        if first_char in self.state_codes:
            # Una confianza alta si el primer carácter es un código de estado válido
            return self.state_codes[first_char], 0.9 
        
        # Manejar caso específico de CDMX si es un patrón reconocido (ej. 'CDMX' explícito o algo similar)
        # Esto requeriría patrones de placa que incluyan 'CDMX' o una lógica más compleja
        if plate_text.startswith('CDMX') and 'CDMX' in self.state_codes:
            return self.state_codes['CDMX'], 0.95
        
        return "No detectado", 0.0

    def process_image(self, image_path):
        """Procesamiento completo de una imagen"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"No se pudo leer la imagen: {image_path}. Verifique la ruta y el archivo.")
                return None, None, None, 0.0
            
            # Paso 1: Detección de candidatos a placa
            candidates = self.detect_plate_candidates(image)
            if not candidates:
                # print(f"No se encontraron candidatos a placa en: {image_path}")
                return None, None, None, 0.0
            
            # Procesar hasta 3 mejores candidatos
            results = []
            for candidate in candidates[:3]: # Considerar solo los 3 mejores por puntuación
                try:
                    # Extraer y rectificar placa
                    plate_img = self.extract_plate_region(image, candidate)
                    
                    # Asegurarse de que la imagen de la placa extraída no esté vacía
                    if plate_img is None or plate_img.size == 0:
                        continue

                    # Reconocimiento con confianza
                    plate_text, confidence = self.recognize_with_confidence(plate_img)
                    if not plate_text:
                        continue
                    
                    # Detección de estado
                    state, state_conf = self.detect_state_from_plate(plate_text)
                    
                    # Calcular la confianza total, dando más peso a la confianza del texto reconocido
                    total_conf = (confidence * 0.7 + state_conf * 0.1) + (candidate['score'] * 0.2) # Factor de score del candidato
                    
                    results.append({
                        'plate_text': plate_text,
                        'state': state,
                        'plate_image': plate_img,
                        'confidence': total_conf,
                        'bbox': candidate['bbox']
                    })
                except Exception as e:
                    print(f"Error procesando un candidato en {image_path}: {str(e)}")
                    continue
            
            if not results:
                # print(f"Ningún candidato pudo ser procesado exitosamente en: {image_path}")
                return None, None, None, 0.0
            
            # Seleccionar resultado con mayor confianza
            best_result = max(results, key=lambda x: x['confidence'])
            return (best_result['plate_text'], best_result['state'],
                    best_result['plate_image'], best_result['confidence'])
        except Exception as e:
            print(f"Error crítico procesando {image_path}: {str(e)}")
            return None, None, None, 0.0

def main():
    recognizer = RobustPlateRecognizer()
    
    # Configuración de carpetas
    input_folder = r'C:\Users\Uggly Boy\Downloads\Placas'
    output_folder = 'placas_resultados_mejorado'
    os.makedirs(output_folder, exist_ok=True)
    
    # Procesar todas las imágenes
    for filename in os.listdir(input_folder):
        # Asegurarse de procesar solo archivos de imagen
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            print(f"\nProcesando: {filename}")
            plate_text, state, plate_img, confidence = recognizer.process_image(image_path)
            
            # Guardar resultados
            base_name = os.path.splitext(filename)[0]
            result_file = os.path.join(output_folder, f"{base_name}.txt")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Placa: {plate_text if plate_text else 'No detectada'}\n")
                f.write(f"Estado: {state if state else 'No detectado'}\n")
                f.write(f"Confianza: {confidence:.2f}\n")
            
            # Guardar imagen de la placa recortada si se detectó
            if plate_img is not None and isinstance(plate_img, np.ndarray) and plate_img.size > 0:
                plate_img_path = os.path.join(output_folder, f"{base_name}_plate.jpg")
                try:
                    cv2.imwrite(plate_img_path, plate_img)
                except Exception as e:
                    print(f"Error al guardar la imagen de la placa recortada {plate_img_path}: {str(e)}")
            
            print(f"Resultado final: {plate_text if plate_text else 'No detectada'} ({state if state else 'No detectado'}) [Conf: {confidence:.2f}]")

if __name__ == "__main__":
    main()