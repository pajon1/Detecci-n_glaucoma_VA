import glob
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.measure import label, regionprops

def normalizar_imagen(imagen, umbral=200, gamma=2.0):
    """
    Normaliza una imagen en escala de grises de modo que los píxeles con valores
    entre 'umbral' y 255 se escalen a 0-255 de forma exponencial (transformación gamma),
    y los píxeles por debajo del umbral se establezcan en 0.

    Parámetros:
    - imagen: numpy.ndarray
        Imagen en escala de grises a normalizar.
    - umbral: int, opcional
        Valor de umbral (por defecto es 200).
    - gamma: float, opcional
        Exponente para la transformación gamma (por defecto 2.0).

    Retorna:
    - imagen_normalizada: numpy.ndarray
        Imagen normalizada según los criterios especificados.
    """
    # Asegurarse de que la imagen esté en formato de punto flotante
    imagen = imagen.astype(np.float32)

    # Crear una máscara donde los píxeles son mayores o iguales al umbral
    mascara = imagen >= umbral

    # Inicializar la imagen normalizada con ceros
    imagen_normalizada = np.zeros_like(imagen, dtype=np.float32)

    # Escalamos la región [umbral, 255] a [0, 1]
    # x = (pixel - umbral) / (255 - umbral)
    x = (imagen[mascara] - umbral) / (255.0 - umbral)

    # Aplicamos la transformación exponencial (gamma)
    # y luego reescalamos a [0, 255].
    imagen_normalizada[mascara] = 255.0 * (x ** gamma)

    # Asegurarse de que los valores estén dentro del rango [0, 255]
    imagen_normalizada = np.clip(imagen_normalizada, 0, 255)

    # Convertir la imagen de vuelta a tipo uint8
    imagen_normalizada = imagen_normalizada.astype(np.uint8)

    return imagen_normalizada


import numpy as np
from skimage.transform import hough_circle
from skimage.draw import circle_perimeter

def get_circle_border_pixels(roi_closed,center_x,center_y):
    """
    Identifica el círculo "óptimo" dentro de la ROI (suponiendo que el centro sea la mitad
    del ancho y alto de la ROI) usando la Transformada de Hough y devuelve las coordenadas
    (x, y) de los píxeles del perímetro de ese círculo.

    Parámetros:
    -----------
    roi_closed : np.ndarray
        Imagen (ROI) 2D, puede ser binaria o de niveles de gris.

    Retorna:
    --------
    border_pixels : np.ndarray
        Arreglo Nx2 con las coordenadas (x, y) de los píxeles del perímetro.
    center_and_radius : tuple
        Tupla (cx, cy, r) con el centro y radio óptimos según la Hough Circle.
    """

    # 1) Normalizar la ROI en [0..1] (opcional, según tu flujo)
    roi_float = roi_closed.astype(np.float32)
    max_val_roi = roi_float.max()
    if max_val_roi > 0:
        roi_normalized = roi_float / (max_val_roi + 1e-6)
    else:
        roi_normalized = roi_float

    # 2) Definir centro de la ROI
    roi_h, roi_w = roi_closed.shape
    roi_center_x = center_x
    roi_center_y = center_y

    # 3) Detectar bordes con Canny
    #edges = canny(roi_closed, sigma=2)
    edges=roi_closed
    # También podrías usar `canny(roi_normalized, sigma=2)` si quieres usar la normalizada.

    # 4) Definir rango de radios de interés (ajusta según tu problema)
    r_min = int(min(roi_h, roi_w) * 0.2)
    r_max = int(min(roi_h, roi_w) * 0.9)
    hough_radii = np.arange(r_min, r_max, 1)

    # 5) Calcular la Transformada de Hough para cada radio
    hough_res = hough_circle(edges, hough_radii)

    # 6) Forzar el centro (roi_center_x, roi_center_y) y buscar el radio con mayor acumulador
    #    - La dimensión 0 de hough_res es el índice de los radios en hough_radii
    #    - La dimensión 1 y 2 corresponden a la posición (y, x) en la imagen
    acc_vals = [hough_res[i, roi_center_y, roi_center_x] for i in range(len(hough_radii))]

    best_idx = np.argmax(acc_vals)
    best_radius = hough_radii[best_idx]
    best_acc = acc_vals[best_idx]  # (por si quieres usarlo)

    # 7) Usando circle_perimeter para obtener los píxeles del perímetro del círculo detectado
    rr, cc = circle_perimeter(roi_center_y, roi_center_x, best_radius)

    # Ojo: rr (row) es equivalente a la coordenada y, y cc (col) es la coordenada x
    # Para unificar en el formato [x, y], hacemos:
    border_pixels = np.column_stack((cc, rr))

    # Por si quieres retornar también el centro y radio
    center_and_radius = (roi_center_x, roi_center_y, best_radius)

    return border_pixels

def get_disc(ruta_imagen, radio=90,mostrar=False):
    """
    Lee una imagen de retinografía, la procesa para diagnosticar glaucoma:
    1) Escala de grises.
    2) Detecta el píxel más brillante (disco óptico).
    3) Extrae una ROI circular (radio=150) alrededor de ese píxel.
    4) Aplica una operación de apertura (kernel) para eliminar venas.
    5) Aplica igualación de histograma.
    6) Define un contorno activo (snake) con mayor 'beta' para mantener forma circular (solo contorno interno).
    7) Genera la máscara binaria correspondiente al contorno interno.
    8) Visualiza resultados con cv2.imshow.

    Parámetros:
    -----------
    ruta_imagen : str
        Ruta a la imagen de entrada (formato compatible con OpenCV).
    radio : int, opcional
        Radio (en píxeles) del recorte circular para la ROI. Por defecto 150.

    Retorna:
    --------
    None (muestra ventanas con cv2.imshow).
    """

    # 1) Lectura de la imagen
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen en {ruta_imagen}.")

    # Mostrar imagen original
    if mostrar==True:
        cv2.imshow("1) Imagen original (BGR)", img_bgr)

    # 2) Convertir a escala de grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mostrar == True:
        cv2.imshow("2) Escala de grises", gray)


    # 3) Encontrar el píxel más brillante (probable centro del disco óptico)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    center_x, center_y = maxLoc  # maxLoc = (x, y)

    # Visualizamos la ubicación del píxel más brillante
    temp_bright = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(temp_bright, (center_x, center_y), 5, (0, 0, 255), -1)
    if mostrar == True:
        cv2.imshow("2.1) Pixel mas brillante (rojo)", temp_bright)

    # 4) Extraer ROI cuadrado

    # Definimos los límites de recorte (bounding box) como un cuadrado
    y_min = max(center_y - radio, 0)
    y_max = min(center_y + radio, gray.shape[0])
    x_min = max(center_x - radio, 0)
    x_max = min(center_x + radio, gray.shape[1])

    # Recortamos la ROI cuadrada directamente sin aplicar máscara
    roi_gray = gray[y_min:y_max, x_min:x_max].copy()


    # Mostramos la ROI cuadrada en escala de grises
    if mostrar == True:
        cv2.imshow("3) ROI recortada en escala de grises (Cuadrado)", roi_gray)

    # 6) Operación de apertura
    kernel = np.ones((30,30), np.uint8)
    roi_closed = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
    roi_closed = cv2.equalizeHist(roi_closed)
    kernel = np.ones((20, 20), np.uint8)
    roi_closed = cv2.morphologyEx(roi_closed, cv2.MORPH_OPEN, kernel)

    roi_Canny = cv2.Canny(roi_closed,10, 180)
    roi_Canny = cv2.GaussianBlur(roi_Canny, (0, 0), sigmaX=2, sigmaY=1)


    #roi_gray = roi_Canny


    #roi_closed = normalizar_imagen(roi_closed, 100)

    if mostrar == True:
        cv2.imshow(f"5) ROI tras cierrre, apertura y ecualizaicon en {ruta_imagen}", roi_closed)
        cv2.imshow(f"6) ROI tras canny en {ruta_imagen}", roi_Canny)
    roi_closed=roi_Canny


    # 7) Definimos un contorno “deformable” (EXTERNO) usando un snake
    #    a) Normalizamos la ROI a [0..1]
    roi_float = roi_closed.astype(np.float32)
    max_val_roi = roi_float.max()
    if max_val_roi > 0:
        roi_normalized = roi_float / (max_val_roi + 1e-6)
    else:
        roi_normalized = roi_float

    #    b) Creamos las coordenadas iniciales para el contorno EXTERNO (círculo grande)
    n_points = 200
    theta = np.linspace(0, 2 * np.pi, n_points)

    roi_h, roi_w = roi_closed.shape
    roi_center_x = roi_w // 2-10
    roi_center_y = roi_h // 2+5

    # Radio inicial ~ 0.8 del radio ROI para un contorno externo
    outer_radius = min(roi_center_x, roi_center_y) * 0.8
    init_outer_r = roi_center_y + outer_radius * np.sin(theta)
    init_outer_c = roi_center_x + outer_radius * np.cos(theta)
    init_outer = np.array([init_outer_r, init_outer_c]).T  # Coordenadas (r, c)

    #    c) (OPCIONAL) Filtramos la ROI con mediana para suavizar ruido
    #roi_median = median(roi_normalized, disk(3))  # Ajusta el tamaño del disco según tu caso

    #    d) Ajustamos el contorno usando active_contour
    #       NOTA: Si tu ROI tiene bordes claros sobre fondo oscuro, podrías usar w_line=0, w_edge=1
    #             y si es al revés, tal vez invertir la imagen o ajustar w_line, w_edge.
    snake_outer = active_contour(
        roi_closed,  # Usa la imagen de bordes
        init_outer,
        alpha=0.4,  # Aumentado para mayor tensión
        beta=0.8,  # Aumentado para mayor rigidez
        w_line=0.1,  # Ignora intensidades de línea
        w_edge=1,  # Enfoca en los bordes
        gamma=0.3,
        boundary_condition='periodic'
    )


    # 8) Visualización y creación de máscara binaria
    #    a) Mostramos la ROI con el contorno final dibujado en azul
    roi_contours_3ch = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    fill_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    for (r, c) in snake_outer:
        rr, cc = int(r), int(c)
        if 0 <= rr < radio * 2 and 0 <= cc < radio * 2:
            roi_contours_3ch[rr, cc] = (255, 0, 0)  # azul (B, G, R)
            fill_mask[rr][cc] = 1

    filas, columnas = np.where(fill_mask == 1)
    fill_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    coordenadas = list(zip(columnas, filas))
    coordenadas = [list(coord) for coord in coordenadas]

    # 3. Convertir las coordenadas a un array de NumPy
    pts = np.array(coordenadas, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(fill_mask, [pts], color=255)

    if mostrar == True:
        cv2.imshow(f"6) ROI + contorno final (azul=outer) en {ruta_imagen}", roi_contours_3ch)



    kernel = np.ones((5, 5), np.uint8)
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel)
    distance_transform = cv2.distanceTransform(fill_mask, cv2.DIST_L2, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_transform)
    centroid_x, centroid_y = max_loc
    coordenadas = np.argwhere(fill_mask == 255)
    centroid_y = int(np.average(coordenadas[:, 0]))
    centroid_x = int(np.average(coordenadas[:, 1]))
    #cv2.circle(fill_mask, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

    if mostrar == True:
        cv2.imshow("7) Mascara externa (outer)", fill_mask)

    # Esperar a que se presione alguna tecla para cerrar
    if mostrar == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Paso 3: Calcular las Distancias desde el Centroide a cada Punto del Contorno
    distances = np.sqrt((coordenadas[:, 1] - centroid_x) ** 2 + (coordenadas[:, 0] - centroid_y) ** 2)

    # Paso 4: Obtener las Distancias Máxima y Mínima
    max_distance = np.max(distances)

    # Aplicar el desplazamiento
    desplazamiento_y = center_x - radio
    desplazamiento_x = center_y - radio

    # Desplazar los puntos
    snake_outer = snake_outer + np.array([desplazamiento_x, desplazamiento_y])



    return int(centroid_x+center_x-radio),int(centroid_y+center_y-radio), max_distance,snake_outer



def get_cup(ruta_imagen, center_x,center_y,radio=80,mostrar=False):
    """
    Lee una imagen de retinografía, la procesa para diagnosticar glaucoma:
    1) Escala de grises.
    2) Detecta el píxel más brillante (disco óptico).
    3) Extrae una ROI circular (radio=150) alrededor de ese píxel.
    4) Aplica una operación de apertura (kernel) para eliminar venas.
    5) Aplica igualación de histograma.
    6) Define un contorno activo (snake) con mayor 'beta' para mantener forma circular (solo contorno interno).
    7) Genera la máscara binaria correspondiente al contorno interno.
    8) Visualiza resultados con cv2.imshow.

    Parámetros:
    -----------
    ruta_imagen : str
        Ruta a la imagen de entrada (formato compatible con OpenCV).
    radio : int, opcional
        Radio (en píxeles) del recorte circular para la ROI. Por defecto 150.

    Retorna:
    --------
    None (muestra ventanas con cv2.imshow).
    """

    # 1) Lectura de la imagen
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen en {ruta_imagen}.")

    # Mostrar imagen original
    if mostrar == True:
        cv2.imshow("1) Imagen original (BGR)", img_bgr)

    # 2) Convertir a escala de grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mostrar == True:
        cv2.imshow("2) Escala de grises", gray)

    # 3) Mostrar el píxel más brillante (probable centro del disco óptico)
    # Visualizamos la ubicación del píxel más brillante
    temp_bright = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    #center_x, center_y = maxLoc  # maxLoc = (x, y)
    cv2.circle(temp_bright, (center_x, center_y), 5, (0, 0, 255), -1)
    if mostrar == True:
        cv2.imshow("2.1) Pixel mas brillante (rojo)", temp_bright)

    # 4) Extraer ROI cuadrado

    # Definimos los límites de recorte (bounding box) como un cuadrado
    y_min = max(center_y - radio, 0)
    y_max = min(center_y + radio, gray.shape[0])
    x_min = max(center_x - radio, 0)
    x_max = min(center_x + radio, gray.shape[1])

    # Recortamos la ROI cuadrada directamente sin aplicar máscara
    roi_gray = gray[y_min:y_max, x_min:x_max].copy()


    # Mostramos la ROI cuadrada en escala de grises
    if mostrar == True:
        cv2.imshow("3) ROI recortada en escala de grises (Cuadrado)", roi_gray)

    # 6) Operación de apertura
    diametro = 80
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diametro, diametro))

    roi_closed = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
    roi_closed = cv2.equalizeHist(roi_closed)
    roi_closed = normalizar_imagen(roi_closed, 160)
    diametro = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diametro, diametro))
    roi_closed = cv2.morphologyEx(roi_closed, cv2.MORPH_OPEN, kernel)



    '''
    kernel = np.ones((20, 20), np.uint8)
    # roi_closed = cv2.equalizeHist(roi_gray)
    #
    roi_closed = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
    roi_closed = cv2.equalizeHist(roi_closed)
    # roi_closed = cv2.GaussianBlur(roi_closed, (0, 0), sigmaX=2, sigmaY=2)
    # roi_closed = cv2.Canny(roi_closed, 10, 220)
    # roi_closed = normalizar_imagen(roi_closed, 200)
    roi_closed = cv2.morphologyEx(roi_closed, cv2.MORPH_CLOSE, kernel)
    '''

    roi_Canny = cv2.Canny(roi_closed, 100, 160)
    roi_Canny = cv2.GaussianBlur(roi_Canny, (0, 0), sigmaX=2, sigmaY=1)


    if mostrar == True:
        cv2.imshow(f"5) ROI tras cierrre y ecualizaicon en {ruta_imagen}", roi_closed)
        cv2.imshow(f"6) ROI tras canny en {ruta_imagen}", roi_Canny)
    #roi_gray = roi_Canny
    roi_closed = roi_Canny


    # 7) Definimos un contorno “deformable” (EXTERNO) usando un snake
    #    a) Normalizamos la ROI a [0..1]
    roi_float = roi_closed.astype(np.float32)
    max_val_roi = roi_float.max()
    if max_val_roi > 0:
        roi_normalized = roi_float / (max_val_roi + 1e-6)
    else:
        roi_normalized = roi_float

    #    b) Creamos las coordenadas iniciales para el contorno EXTERNO (círculo grande)
    n_points = 200
    theta = np.linspace(0, 2 * np.pi, n_points)

    roi_h, roi_w = roi_closed.shape
    roi_center_x = roi_w // 2
    roi_center_y = roi_h // 2

    # Radio inicial ~ 0.8 del radio ROI para un contorno externo
    outer_radius = min(roi_center_x, roi_center_y)
    init_outer_r = roi_center_y + outer_radius * np.sin(theta)
    init_outer_c = roi_center_x + outer_radius * np.cos(theta)
    init_outer = np.array([init_outer_r, init_outer_c]).T  # Coordenadas (r, c)

    #    c) (OPCIONAL) Filtramos la ROI con mediana para suavizar ruido
    # roi_median = median(roi_normalized, disk(3))  # Ajusta el tamaño del disco según tu caso

    #    d) Ajustamos el contorno usando active_contour
    #       NOTA: Si tu ROI tiene bordes claros sobre fondo oscuro, podrías usar w_line=0, w_edge=1
    #             y si es al revés, tal vez invertir la imagen o ajustar w_line, w_edge.
    snake_inner = active_contour(
        roi_closed,  # Usa la imagen de bordes
        init_outer,
        alpha=0.8,  # Aumentado para mayor tensión
        beta=4,  # Aumentado para mayor rigidez
        w_line=0.4,  # Ignora intensidades de línea
        w_edge=1,  # Enfoca en los bordes
        gamma=0.3,
        boundary_condition='periodic'
    )


    # 8) Visualización y creación de máscara binaria
    #    a) Mostramos la ROI con el contorno final dibujado en azul
    roi_contours_3ch = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    fill_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    for (r, c) in snake_inner:
        rr, cc = int(r), int(c)
        if 0 <= rr < radio * 2 and 0 <= cc < radio * 2:
            roi_contours_3ch[rr, cc] = (255, 0, 0)  # azul (B, G, R)
            fill_mask[rr][cc] = 1

    filas, columnas = np.where(fill_mask == 1)
    fill_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    coordenadas = list(zip(columnas, filas))
    coordenadas = [list(coord) for coord in coordenadas]

    # 3. Convertir las coordenadas a un array de NumPy
    pts = np.array(coordenadas, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(fill_mask, [pts], color=255)

    if mostrar == True:
        cv2.imshow(f"6) ROI + contorno final (azul=outer) en {ruta_imagen}", roi_contours_3ch)

    kernel = np.ones((5, 5), np.uint8)
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel)
    distance_transform = cv2.distanceTransform(fill_mask, cv2.DIST_L2, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_transform)

    centroid_x, centroid_y = max_loc

    coordenadas = np.argwhere(fill_mask == 255)
    centroid_y= int(np.average( coordenadas[:, 0]))
    centroid_x = int(np.average(coordenadas[:, 1]))
    '''
    fill_mask[centroid_x, centroid_y]=255
    fill_mask[centroid_x+1, centroid_y] = 255
    fill_mask[centroid_x-1, centroid_y] = 255
    fill_mask[centroid_x, centroid_y+1] = 255
    fill_mask[centroid_x, centroid_y - 1] = 255
    '''
    #cv2.circle(fill_mask, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
    if mostrar == True:
        cv2.imshow("7) Mascara externa (outer)", fill_mask)

    '''
    snake_inner=get_circle_border_pixels(fill_mask,centroid_x,centroid_y)

    roi_contours_3ch = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    fill_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    for (r, c) in snake_inner:
        rr, cc = int(r), int(c)
        if 0 <= rr < radio * 2 and 0 <= cc < radio * 2:
            roi_contours_3ch[rr, cc] = (255, 0, 0)  # azul (B, G, R)
            fill_mask[rr][cc] = 1

    filas, columnas = np.where(fill_mask == 1)
    fill_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    coordenadas = list(zip(columnas, filas))
    coordenadas = [list(coord) for coord in coordenadas]

    # 3. Convertir las coordenadas a un array de NumPy
    pts = np.array(coordenadas, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(fill_mask, [pts], color=255)

    roi_contours_3ch[centroid_x,centroid_y]=(0, 255, 0)

    if mostrar == True:
        cv2.imshow(f"8) ROI + contorno final (azul=outer) en {ruta_imagen} tras Hougth", roi_contours_3ch)


    if mostrar == True:
        cv2.imshow("9) Mascara externa (outer) tras Hougth", fill_mask)

    '''

    # Esperar a que se presione alguna tecla para cerrar
    if mostrar == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Paso 3: Calcular las Distancias desde el Centroide a cada Punto del Contorno
    distances = np.sqrt((coordenadas[:, 1] - centroid_x) ** 2 + (coordenadas[:, 0] - centroid_y) ** 2)

    # Paso 4: Obtener las Distancias Máxima y Mínima
    max_distance = np.max(distances)

    desplazamiento_y = center_x - radio
    desplazamiento_x = center_y - radio

    # Desplazar los puntos
    snake_inner = snake_inner + np.array([desplazamiento_x, desplazamiento_y])

    return snake_inner,max_distance,int(centroid_x+center_x-radio),int(centroid_y+center_y-radio)






def mostrar_imagen_con_circulos(imagen,direccion_imagen,snake_outer,snake_inner):
    """
    Carga una imagen desde la dirección proporcionada, dibuja dos círculos en las posiciones y radios especificados, y muestra la imagen.

    Parámetros:
    - direccion_imagen (str): Ruta de la imagen a cargar.
    - posicion1 (tuple): Coordenadas (x, y) del centro del primer círculo.
    - radio1 (int): Radio del primer círculo.
    - posicion2 (tuple): Coordenadas (x, y) del centro del segundo círculo.
    - radio2 (int): Radio del segundo círculo.
    - color1 (tuple): Color del primer círculo en BGR (por defecto rojo).
    - color2 (tuple): Color del segundo círculo en BGR (por defecto azul).
    - grosor (int): Grosor de las líneas de los círculos. Usa -1 para círculos llenos.
    """


    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen desde la dirección '{direccion_imagen}'. Verifica la ruta y el nombre del archivo.")
        return

    (x,y,dim)=imagen.shape
    fill_mask_outer = np.zeros((x, y), dtype=np.uint8)
    for (r, c) in snake_outer:
        rr, cc = int(r), int(c)
        if 0 <= rr < x  and 0 <= cc < y:
            imagen[rr, cc] = (255, 0, 0)  # azul (B, G, R)
            fill_mask_outer[rr][cc] = 1

    filas, columnas = np.where(fill_mask_outer == 1)
    fill_mask_outer = np.zeros((x, y), dtype=np.uint8)
    coordenadas = list(zip(columnas, filas))
    coordenadas = [list(coord) for coord in coordenadas]

    # 3. Convertir las coordenadas a un array de NumPy
    pts = np.array(coordenadas, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(fill_mask_outer, [pts], color=255)


    fill_mask_inner = np.zeros((x, y), dtype=np.uint8)
    for (r, c) in snake_inner:
        rr, cc = int(r), int(c)
        if 0 <= rr < x and 0 <= cc < y:
            imagen[rr, cc] = (0, 0, 255)  # azul (B, G, R)
            fill_mask_inner[rr][cc] = 1

    filas, columnas = np.where(fill_mask_inner == 1)
    fill_mask_inner = np.zeros((x, y), dtype=np.uint8)
    coordenadas = list(zip(columnas, filas))
    coordenadas = [list(coord) for coord in coordenadas]

    # 3. Convertir las coordenadas a un array de NumPy
    pts = np.array(coordenadas, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(fill_mask_inner, [pts], color=255)

    kernel = np.ones((3, 3), np.uint8)
    fill_mask_inner = cv2.morphologyEx(fill_mask_inner, cv2.MORPH_CLOSE, kernel)
    fill_mask_outer = cv2.morphologyEx(fill_mask_outer, cv2.MORPH_CLOSE, kernel)


    # Mostrar la imagen con los círculos dibujados
    cv2.imshow('Imagen con Circulos de '+direccion_imagen,imagen)
    #cv2.imshow('inner '+direccion_imagen,fill_mask_inner)
    #cv2.imshow('outer ' + direccion_imagen, fill_mask_outer)
    return fill_mask_outer,fill_mask_inner


def procesar_carpeta(ruta_carpeta):
    """
    Procesa todas las imágenes en la carpeta especificada, detecta el disco y la copa,
    y muestra cada imagen con los círculos dibujados.

    :param ruta_carpeta: Ruta a la carpeta que contiene las imágenes.
    """
    # Definir los formatos de imagen que deseas procesar
    formatos = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')

    # Crear una lista vacía para almacenar las rutas de las imágenes
    rutas_imagenes = []

    # Iterar sobre cada formato y agregar las rutas de las imágenes a la lista
    for formato in formatos:
        rutas_imagenes.extend(glob.glob(os.path.join(ruta_carpeta, formato)))

    if not rutas_imagenes:
        print(f"No se encontraron imágenes en la carpeta: {ruta_carpeta}")
        return

    # Procesar cada imagen
    for ruta_imagen in rutas_imagenes:
        try:
            # Obtener las coordenadas y radio del disco
            center_x_disc, center_y_disc, radio_disc,snake_disc = get_disc(ruta_imagen, mostrar=False)

            # Obtener las coordenadas y radio de la copa
            radio_cup = int(radio_disc)  # Ajustar según sea necesario
            snake_cup,radio_cup,center_x_cup,center_y_cup = get_cup(ruta_imagen,center_x_disc, center_y_disc ,mostrar=False,radio=radio_cup+10
            )
            # Cargar la imagen
            imagen = cv2.imread(ruta_imagen)
            cv2.circle(imagen, (center_x_cup, center_y_cup), 5, (0, 255, 0), -1)
            cv2.circle(imagen, (center_x_disc, center_y_disc), 5, (0, 0, 0), -1)
            # Mostrar la imagen con los círculos dibujados
            fill_mask_outer,fill_mask_inner=mostrar_imagen_con_circulos(
                imagen,
                ruta_imagen,
                snake_disc,
                snake_cup
            )


            print(f"Procesada: {Path(ruta_imagen).name}")
            print(f"Ratio Copa-Disco (CDR): {radio_cup/radio_disc:.2f}")

        except Exception as e:
            print(f"Error al procesar {ruta_imagen}: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    procesar_carpeta("./imagenes")
    '''
    ruta_imagen = "imagenes/n0005.png"  # Ajusta la ruta a tu imagen
    # Obtener las coordenadas y radio del disco
    center_x_disc, center_y_disc, radio_disc, snake_disc = get_disc(ruta_imagen, mostrar=False)

    # Obtener las coordenadas y radio de la copa
    radio_cup = int(radio_disc)  # Ajustar según sea necesario
    snake_cup, max_distance_cup,center_x_cup, center_y_cup = get_cup(ruta_imagen, center_x_disc, center_y_disc, mostrar=True, radio=radio_cup+10)
    imagen = cv2.imread(ruta_imagen)
    cv2.circle(imagen, (center_x_cup, center_y_cup), 5, (0, 255, 0), -1)
    cv2.circle(imagen, (center_x_disc, center_y_disc), 5, (0, 0, 0), -1)
    fill_mask_outer, fill_mask_inner = mostrar_imagen_con_circulos(
        imagen,
        ruta_imagen,
        snake_disc,
        snake_cup
    )
    '''



    cv2.waitKey(0)
    cv2.destroyAllWindows()
