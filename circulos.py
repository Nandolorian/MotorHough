import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from collections import defaultdict


# Se lee y se muestra la imagen a procesar para detectar las líneas.
img = cv2.imread('./motor.png')
plt.figure(facecolor="#E8E4E4",figsize=(6, 6)).canvas.set_window_title("Imagen a procesar")
plt.imshow(img)
plt.show()

# Se remueve el color de la imagen
imgGris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Se hace un blur para mejorar la detección de bordes
imgBlur = cv2.GaussianBlur(imgGris, (5, 5), 1.5)
bordes = cv2.Canny(imgGris, 250, 400)

# Se muestra la imagen procesada
plt.figure(facecolor="#E8E4E4",figsize=(6, 6)).canvas.set_window_title("Bordes detectados")
plt.imshow(bordes)
plt.show()

# Tomamos las medidas de la imagen
rango_y, rango_x = bordes.shape

# Se calcula el rango de Theta de 0 a 360 grados con incrementos de 3
rango_theta = np.arange(0, 360, 3)

# Se calculan los radios min y máx a buscar. En el caso de estudio el radio es fijo y para la imagen del motor se usa el valor 30
rango_r = np.arange(30, 31, step=1)

# Se precalculan los posibles circulos según los radios y thetas dados utilizando la ecuación de la circunferencia
c_detectados = []
for radio in rango_r:
    for theta in range(100):
        c_detectados.append((radio, int(radio*math.cos(2*math.pi*theta/100)), int(radio*math.sin(2*math.pi*theta/100))))

# Creamos el acumulador usando las dimensiones de la imagen
acumulador = defaultdict(int)
for y in range(rango_y):
    for x in range(rango_x):
        if bordes[y][x] != 0: 
            for r, rcos_t, rsin_t in c_detectados:
                x_center = x - rcos_t
                y_center = y - rsin_t
                acumulador[(x_center, y_center, r)] += 1

# Con el acumulador armado lo vamos recorriendo de forma ordenada 
# según los máximos detectados usando un límite para los máximos locales
limite = 0.50
circulos = []
for circulo, max in sorted(acumulador.items(), key=lambda i: -i[1]):
    x, y, r = circulo
    local = max / 100
    if local >= limite: 
        circulos.append((x, y, r, local))

#####
for x, y, r, v in circulos:
    print(x,y,r,v)

#####

# Se toma la imagen original para graficar sobre ella los circulos detectados
imgFinal = img.copy()

for x, y, r, v in circulos:
    imgFinal = cv2.circle(imgFinal, (x,y), r, (255,0,0), 2)

plt.figure(facecolor="#E8E4E4",figsize=(6, 6)).canvas.set_window_title("Circulos detectados")
plt.imshow(imgFinal)
plt.show()