#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[17]:



print ("Programa de predicción version 1.0 ")
print (" ")
print ("Observa las opciones y elige una.")
print ("1.- Opcion 1 Mostrar el arreglo completo.")
print ("2.- Opcion 2 Agregar un numero nuevo al array ya existente.")
print ("3.- Opcion 3 Mostrar la predicción con nuestro Algoritmo.")
print ("4.- Opcion 4 Mostrar la predicción sin el Algoritmo")
print ("5.- Opcion 5 Mostrar la predicción  de ambas listas")
print ("6.- Opcion 6 Comparacion de efectividad entre algoritmos")
print ("7.- Opcion 7 Informacion general del arreglo.")
print ("8.- Opcion 8 Eleccion de Slices o Rebanadas.")
print ("9.- Opcion 9 Break")
print (" ")

opcion=int(input("¿Qué opción deseas ingresar? "))
print (" ")
opcionbaja=0
opcionlimite=11

if opcion > 0 and opcion <opcionlimite:
    print ("Eligió una opción correcta, Eligió la opción ",opcion)
else:
    print("La opción NO está dentro de los parametros establecidos")
    
if opcion == 1:
    print ("")
    print ("*****Estas son nuestras matrices*****")
    
    lote ={'Valor A' :pd.Series ([3,5,5,9,1,5,6,9,5,4,1,3,4,5,1,1,3,4,9,9,9,3,1,8,
                             2,6,5,1,3,12,2,5,4,4,3,13,4,1,1,5,2,7,3,11,5,5,14,
                             4,4,10,1,8,2,4,4,1,1,3,4,5,1,12,#Diciembre del 2020
                              
                             1,2,4,8,4,1,2,6,1,3,6,3,2,3,4,1,3,1,4,8,1,2,
                             6,5,1,1,1,3,1,1,1,1,1,1,2,1,9,3,4,3,8,7,4,3,11,
                             6,3,4,3,4,3,1,3,1,2,1,1,1,3,3,#Noviembre 2020
                              
                             8,1,2,1,4,4,12,3,6,7,3,1,6,8,4,4,7,2,4,7,4,3,6,11,
                             6,1,3,5,3,2,5,4,4,2,1,1,3,2,5,4,4,4,1,4,8,3,3,5,6,3,5,9,
                             2,5,4,1,1,3,2,3,6,1, #Enero 2021
                              
                             5,10,5,5,1,4,3,6,2,4,3,1,4,3,10,4,3,2,3,9,10,6,4,
                             7,1,1,9,1,11,1,8,6,9,5,3,12,1,5,5,2,2,6,6,4,5,5,9,3,
                             4,1,1,3,17,2,5,4, #Febrero 2021
                              
                             3,5,1,7,5,4,14,2,19,1,2,7,5,5,5,10,12,3,10,9,2,5,3,2,2,5,
                             2,12,14,6,7,1,2,3,5,1,4,6,4,10,2,6,5,1,2,4,6,4,5,3,2,6,1,5,11,
                             1,2,2,4,12,1,6, #Marzo 2021
                            
                             10,4,1,12,2,11,1,4,7,2,6,5,5,2,10,3,5,3,5,2,7,5,11,
                             2,3,3,12,2,4,2,9,6,3,5,3,4,1,4,3,2,1,3,1,6,2,2,4,1,2,2,
                             3,1,2,2,6,4,2,2,13,12, #Abril 2021
                              
                             1,1,15,1,2,9,1,9,2,9,1,1,7,16,5,12,6,12,6,3,1,1,1,8,5,2,
                             10,6,2,2,8,3,11,3,1,4,2,10,2,8,5,8,7,4,1,1,13,14,11,10,3,2,2,
                             1,3,3,4,3,6,7,1,1,#mAYO2021
                              
                             9,9,1,6,1,9,1,1,5,8,8,5,7,3,4,9,1,1,1,1,9,3,4,7,5,13,2,10,6,8,9,4,
                              5,2,2,6,1,17,8,6,2,3,3,2,2,4,6,5,8,2,3,2,14,2,4,6,12,10,1,11, #JUNIO 2021
                              
                              16,6,12,15,3,4,3,11,9,8,12,2,1,6,3,3,8,4,10,2,1,3,2,2,12,1,
                              7,5,3,6,2,5,5,7,6,5,5,2,5,3,2,3,2,4,7,11,3,2,2,15,4,1,2,7,4,
                              5,1,1,9,7,3,2, #Julio 2021
                              
                             1,15,2,1,4,4,14,2,10,11,7,5,1,4,11,1,5,12,1,1,10,9,1,2,5,4,
                              7,3,3,4,10,3,6,1,1,1,1,8,4,2,3,3,5,3,10,6,3,2,8,17,10,1,8,4,
                              1,5,5,2,1,3,1,4, #Agosto A
                              
                              6,8,5,11,6,2,3,1,4,1,5,9,8,5,7,3,1,4,2,4,4,2,3,4,4,9,
                              8,2,7,3,6,3,4,3,1,6,7,4,5,5,1,1,4,3,9,8,3,4,8,8,5,12,13,2,
                              2,2,4,1,1,6, #Septiembre A
                              
                              10,3,3,2,2,3,5,6,3,4,10,2,11,2,9,8,6,6,16,3,9,5,3,8,
                              2,3,5,3,4,2,4,4,3,7,7,1,14,6,1,8,1,2,3,3,1,2,6,2,2,4,
                              2,5,6,1,3,1,7,3,1,7,1,2, #October 2021 A
                              
                              5,1,10,9,19,2,1,10,6,7,1,4,13,6,4,2,1,5,1,4,1,5,3,1,4,5,3,4,
                              3,6,2,7,8,7,2,4,12,2,8,12,2,8,9,1,7,1,1,6,7,6,4,4,7,2,1,1,1,
                              3,7,1, #November 2021
                              
                              2,3,1,2,4,3,7,2,6,9,1,10,15,2,1,15,2,6,14,6,15,10,2,5,2,12,
                              5,9,5,3,3,5,4,1,6,1,4,6,3,2,4,5,4,1,1,6,1,7,3,1,1,5,1,3,1,1,
                              2,2,4,11,2,1, #Diciembre 2021 A
                              
                              #Enero 2022
                              1,2,1,3,5,9,5,5,8,1,3,3,10,13,9,2,1,3,5,4
                              
                             
                             ],
                                 index=['Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                    
                                   'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                   
                                    'Enero 31','Enero 31 ','Enero 30','Enero 30 ','Enero 29','Enero 29 ',
                                    'Enero 28','Enero 28 ','Enero 27','Enero 27 ',
                                    'Enero 26','Enero 26 ','Enero 25','Enero 25 ',
                                    'Enero 24','Enero 24 ','Enero 23','Enero 23 ',
                                    'Enero 22','Enero 22 ','Enero 21','Enero 21 ',
                                    'Enero 20','Enero 20 ','Enero 19','Enero 19 ',
                                    'Enero 18','Enero 18 ','Enero 17','Enero 17 ',
                                    'Enero 16','Enero 16 ','Enero 15','Enero 15 ',
                                    'Enero 14','Enero 14 ','Enero 13','Enero 13 ',
                                    'Enero 12','Enero 12 ','Enero 11','Enero 11 ',
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ',
                                    
                                    'Febrero  28','Febrero  28 ','Febrero  27','Febrero  27 ',
                                    'Febrero  26','Febrero  26 ','Febrero  25','Febrero  25 ',
                                    'Febrero  24','Febrero  24 ','Febrero  23','Febrero  23 ',
                                    'Febrero  22','Febrero  22 ','Febrero  21','Febrero  21 ',
                                    'Febrero  20','Febrero  20 ','Febrero  19','Febrero  19 ',
                                    'Febrero  18','Febrero  18 ','Febrero  17','Febrero  17 ',
                                    'Febrero  16','Febrero  16 ','Febrero  15','Febrero  15 ',
                                    'Febrero  14','Febrero  14 ','Febrero  13','Febrero  13 ',
                                    'Febrero  12','Febrero  12 ','Febrero  11','Febrero  11 ',
                                    'Febrero  10','Febrero  10 ','Febrero  9 ','Febrero  9 ',
                                    'Febrero  8 ','Febrero  8  ','Febrero  7 ','Febrero  7 ',
                                    'Febrero  6 ','Febrero  6  ','Febrero  5 ','Febrero  5 ',
                                    'Febrero  4 ','Febrero  4  ','Febrero  3 ','Febrero  3 ',
                                    'Febrero  2 ','Febrero  2  ','Febrero  1 ','Febrero  1 ',
                                    
                                    'Marzo  31','Marzo  31 ', 'Marzo  30',
                                    'Marzo  30 ','Marzo  29','Marzo  29 ',
                                    'Marzo  28','Marzo  28 ','Marzo  27','Marzo  27 ',
                                    'Marzo  26','Marzo  26 ','Marzo  25','Marzo  25 ',
                                    'Marzo  24','Marzo  24 ','Marzo  23','Marzo  23 ',
                                    'Marzo  22','Marzo  22 ','Marzo  21','Marzo  21 ',
                                    'Marzo  20','Marzo  20 ','Marzo  19','Marzo  19 ',
                                    'Marzo  18','Marzo  18 ','Marzo  17','Marzo  17 ',
                                    'Marzo  16','Marzo  16 ','Marzo  15','Marzo  15 ',
                                    'Marzo  14','Marzo  14 ','Marzo  13','Marzo  13 ',
                                    'Marzo  12','Marzo  12 ','Marzo  11','Marzo  11 ',
                                    'Marzo  10','Marzo  10 ','Marzo  9 ','Marzo  9 ',
                                    'Marzo  8 ','Marzo  8  ','Marzo  7 ','Marzo  7 ',
                                    'Marzo  6 ','Marzo  6  ','Marzo  5 ','Marzo  5 ',
                                    'Marzo  4 ','Marzo  4  ','Marzo  3 ','Marzo  3 ',
                                    'Marzo  2 ','Marzo  2  ','Marzo  1 ','Marzo  1 ',
                                    
                                    'Abril  30','Abril  30 ','Abril  29','Abril  29 ',
                                    'Abril  28','Abril  28 ','Abril  27','Abril  27 ',
                                    'Abril  26','Abril  26 ','Abril  25','Abril  25 ',
                                    'Abril  24','Abril  24 ','Abril  23','Abril  23 ',
                                    'Abril  22','Abril  22 ','Abril  21','Abril  21 ',
                                    'Abril  20','Abril  20 ','Abril  19','Abril  19 ',
                                    'Abril  18','Abril  18 ','Abril  17','Abril  17 ',
                                    'Abril  16','Abril  16 ','Abril  15','Abril  15 ',
                                    'Abril  14','Abril  14 ','Abril  13','Abril  13 ',
                                    'Abril  12','Abril  12 ','Abril  11','Abril  11 ',
                                    'Abril  10','Abril  10 ','Abril  9 ','Abril  9 ',
                                    'Abril  8 ','Abril  8  ','Abril  7 ','Abril  7 ',
                                    'Abril  6 ','Abril  6  ','Abril  5 ','Abril  5 ',
                                    'Abril  4 ','Abril  4  ','Abril  3 ','Abril  3 ',
                                    'Abril  2 ','Abril  2  ','Abril  1 ','Abril  1 ',
                                    
                                    'Mayo  31','Mayo  31 ',
                                    'Mayo  30','Mayo  30 ','Mayo  29','Mayo  29 ',
                                    'Mayo  28','Mayo  28 ','Mayo  27','Mayo  27 ',
                                    'Mayo  26','Mayo  26 ','Mayo  25','Mayo  25 ',
                                    'Mayo  24','Mayo  24 ','Mayo  23','Mayo  23 ',
                                    'Mayo  22','Mayo  22 ','Mayo  21','Mayo  21 ',
                                    'Mayo  20','Mayo  20 ','Mayo  19','Mayo  19 ',
                                    'Mayo  18','Mayo  18 ','Mayo  17','Mayo  17 ',
                                    'Mayo  16','Mayo  16 ','Mayo  15','Mayo  15 ',
                                    'Mayo  14','Mayo  14 ','Mayo  13','Mayo  13 ',
                                    'Mayo  12','Mayo  12 ','Mayo  11','Mayo  11 ',
                                    'Mayo  10','Mayo  10 ','Mayo  9 ','Mayo  9 ',
                                    'Mayo  8 ','Mayo  8  ','Mayo  7 ','Mayo  7 ',
                                    'Mayo  6 ','Mayo  6  ','Mayo  5 ','Mayo  5 ',
                                    'Mayo  4 ','Mayo  4  ','Mayo  3 ','Mayo  3 ',
                                    'Mayo  2 ','Mayo  2  ','Mayo  1 ','Mayo  1 ',
                                    
                                    'Junio 30','Junio 30 ','Junio 29','Junio 29 ',
                                    'Junio 28','Junio 28 ','Junio 27','Junio 27 ',
                                    'Junio 26','Junio 26 ','Junio 25','Junio 25 ',
                                    'Junio 24','Junio 24 ','Junio 23','Junio 23 ',
                                    'Junio 22','Junio 22 ','Junio 21','Junio 21 ',
                                    'Junio 20','Junio 20 ','Junio 19','Junio 19 ',
                                    'Junio 18','Junio 18 ','Junio 17','Junio 17 ',
                                    'Junio 16','Junio 16 ','Junio 15','Junio 15 ',
                                    'Junio 14','Junio 14 ','Junio 13','Junio 13 ',
                                    'Junio 12','Junio 12 ','Junio 11','Junio 11 ',
                                    'Junio 10','Junio 10 ','Junio 9 ','Junio 9 ',
                                    'Junio 8 ','Junio 8  ','Junio 7 ','Junio 7 ',
                                    'Junio 6 ','Junio 6  ','Junio 5 ','Junio 5 ',
                                    'Junio 4 ','Junio 4  ','Junio 3 ','Junio 3 ',
                                    'Junio 2 ','Junio 2  ','Junio 1 ','Junio 1 ',
                                    
                                    'Julio 31','Julio 31','Julio 30','Julio 30 ','Julio 29','Julio 29 ',
                                    'Julio 28','Julio 28 ','Julio 27','Julio 27 ',
                                    'Julio 26','Julio 26 ','Julio 25','Julio 25 ',
                                    'Julio 24','Julio 24 ','Julio 23','Julio 23 ',
                                    'Julio 22','Julio 22 ','Julio 21','Julio 21 ',
                                    'Julio 20','Julio 20 ','Julio 19','Julio 19 ',
                                    'Julio 18','Julio 18 ','Julio 17','Julio 17 ',
                                    'Julio 16','Julio 16 ','Julio 15','Julio 15 ',
                                    'Julio 14','Julio 14 ','Julio 13','Julio 13 ',
                                    'Julio 12','Julio 12 ','Julio 11','Julio 11 ',
                                    'Julio 10','Julio 10 ','Julio 9 ','Julio 9 ',
                                    'Julio 8 ','Julio 8  ','Julio 7 ','Julio 7 ',
                                    'Julio 6 ','Julio 6  ','Julio 5 ','Julio 5 ',
                                    'Julio 4 ','Julio 4  ','Julio 3 ','Julio 3 ',
                                    'Julio 2 ','Julio 2  ','Julio 1 ','Julio 1 ',
                                    
                                    'Agosto 31','Agosto 31','Agosto 30','Agosto 30 ','Agosto 29','Agosto 29 ',
                                    'Agosto 28','Agosto 28 ','Agosto 27','Agosto 27 ',
                                    'Agosto 26','Agosto 26 ','Agosto 25','Agosto 25 ',
                                    'Agosto 24','Agosto 24 ','Agosto 23','Agosto 23 ',
                                    'Agosto 22','Agosto 22 ','Agosto 21','Agosto 21 ',
                                    'Agosto 20','Agosto 20 ','Agosto 19','Agosto 19 ',
                                    'Agosto 18','Agosto 18 ','Agosto 17','Agosto 17 ',
                                    'Agosto 16','Agosto 16 ','Agosto 15','Agosto 15 ',
                                    'Agosto 14','Agosto 14 ','Agosto 13','Agosto 13 ',
                                    'Agosto 12','Agosto 12 ','Agosto 11','Agosto 11 ',
                                    'Agosto 10','Agosto 10 ','Agosto 9 ','Agosto 9 ',
                                    'Agosto 8 ','Agosto 8  ','Agosto 7 ','Agosto 7 ',
                                    'Agosto 6 ','Agosto 6  ','Agosto 5 ','Agosto 5 ',
                                    'Agosto 4 ','Agosto 4  ','Agosto 3 ','Agosto 3 ',
                                    'Agosto 2 ','Agosto 2  ','Agosto 1 ','Agosto 1 ',
                                    
                                    'Septiembre 30','Septiembre 30 ','Septiembre 29','Septiembre 29 ',
                                    'Septiembre 28','Septiembre 28 ','Septiembre 27','Septiembre 27 ',
                                    'Septiembre 26','Septiembre 26 ','Septiembre 25','Septiembre 25 ',
                                    'Septiembre 24','Septiembre 24 ','Septiembre 23','Septiembre 23 ',
                                    'Septiembre 22','Septiembre 22 ','Septiembre 21','Septiembre 21 ',
                                    'Septiembre 20','Septiembre 20 ','Septiembre 19','Septiembre 19 ',
                                    'Septiembre 18','Septiembre 18 ','Septiembre 17','Septiembre 17 ',
                                    'Septiembre 16','Septiembre 16 ','Septiembre 15','Septiembre 15 ',
                                    'Septiembre 14','Septiembre 14 ','Septiembre 13','Septiembre 13 ',
                                    'Septiembre 12','Septiembre 12 ','Septiembre 11','Septiembre 11 ',
                                    'Septiembre 10','Septiembre 10 ','Septiembre 9 ','Septiembre 9 ',
                                    'Septiembre 8 ','Septiembre 8  ','Septiembre 7 ','Septiembre 7 ',
                                    'Septiembre 6 ','Septiembre 6  ','Septiembre 5 ','Septiembre 5 ',
                                    'Septiembre 4 ','Septiembre 4  ','Septiembre 3 ','Septiembre 3 ',
                                    'Septiembre 2 ','Septiembre 2  ','Septiembre 1 ','Septiembre 1 ',
                                    
                                    'Octubre 31','Octubre 31','Octubre 30','Octubre 30 ','Octubre 29','Octubre 29 ',
                                    'Octubre 28','Octubre 28 ','Octubre 27','Octubre 27 ',
                                    'Octubre 26','Octubre 26 ','Octubre 25','Octubre 25 ',
                                    'Octubre 24','Octubre 24 ','Octubre 23','Octubre 23 ',
                                    'Octubre 22','Octubre 22 ','Octubre 21','Octubre 21 ',
                                    'Octubre 20','Octubre 20 ','Octubre 19','Octubre 19 ',
                                    'Octubre 18','Octubre 18 ','Octubre 17','Octubre 17 ',
                                    'Octubre 16','Octubre 16 ','Octubre 15','Octubre 15 ',
                                    'Octubre 14','Octubre 14 ','Octubre 13','Octubre 13 ',
                                    'Octubre 12','Octubre 12 ','Octubre 11','Octubre 11 ',
                                    'Octubre 10','Octubre 10 ','Octubre 9 ','Octubre 9 ',
                                    'Octubre 8 ','Octubre 8  ','Octubre 7 ','Octubre 7 ',
                                    'Octubre 6 ','Octubre 6  ','Octubre 5 ','Octubre 5 ',
                                    'Octubre 4 ','Octubre 4  ','Octubre 3 ','Octubre 3 ',
                                    'Octubre 2 ','Octubre 2  ','Octubre 1 ','Octubre 1 ',
                                    
                                    'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                    
                                    'Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                    
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ']),
           'Valor B' :pd.Series ([11,8,13,11,9,8,10,22,19,5,6,5,6,9,5,4,7,11,18,13,20,6,2,9,
                            15,13,10,14,16,14,6,7,12,9,5,16,5,11,5,7,4,9,7,12,
                            8,7,15,8,13,13,8,11,12,5,11,4,2,4,5,7,2,15,#Diciembre 2020
                            
                            6,13,12,12,5,4,4,11,5,9,12,8,4,4,17,5,6,10,9,9,3,9,
                            11,16,8,8,2,4,2,5,4,4,15,13,11,6,11,4,11,10,14,8,20,
                            4,13,12,9,5,5,9,5,3,10,3,17,2,3,3,5,4,#Noviembre 2020
                               
                            16,4,16,3,6,5,20,4,9,11,9,11,13,12,8,5,10,8,5,9,6,5,8,12,11,5,
                            5,6,8,16,13,9,5,8,4,9,8,13,7,6,15,11,15,11,4,9,9,9,10,9,7,20,
                            3,14,5,12,3,4,4,4,12,11, #Enero 2021
                            
                            13,11,16,8,2,13,15,12,9,8,5,2,16,5,14,11,7,4,5,13,12,7,6,10,9,10,15,
                             5,12,9,9,8,10,9,11,13,4,16,13,3,4,15,9,6,6,12,20,13,5,2,4,17,18,6,15,16,  #Febrero2021
                            
                             
                            7,15,8,19,16,8,16,7,21,2,5,10,8,10,8,12,17,5,17,14,3,9,12,8,3,10,
                            12,13,19,4,11,4,20,6,7,4,12,12,13,18,10,10,10,5,4,5,11,10,7,6,4,18,
                            2,6,18,6,3,10,7,17,4,7, #Marzo 2021
                            
                            12,8,3,13,4,15,9,9,11,6,7,6,12,7,17,10,7,6,15,12,12,9,16,5,
                            4,8,16,3,7,4,16,8,20,12,8,11,2,10,8,9,3,5,2,8,4,11,10,2,7,
                            5,9,11,3,21,11,11,5,4,18,13, #Abril 2021
                               
                            5,9,20,13,13,10,6,16,7,12,8,12,8,20,14,20,11,15,7,10,3,3,2,9,6,8,12,12,12,
                            10,10,5,17,4,10,5,9,11,9,10,10,9,16,11,11,11,18,15,13,14,10,4,8,5,4,21,
                            5,10,8,10,9,6, #MAYO 2021
                               
                            13,11,2,6,3,16,2,4,11,11,12,7,11,15,10,16,6,8,13,15,14,8,6,17,7,14,6,
                            11,9,13,19,6,8,10,3,11,4,20,9,14,12,4,8,4,10,9,17,7,17,7,16,5,17,3,5,
                            8,13,20,10,12, #JUNIO 2021
                            
                            19,7,14,20,6,7,7,14,10,18,16,4,8,7,10,7,10,6,14,4,10,7,6,10,13,5,
                            14,7,12,14,5,9,12,12,9,6,7,4,9,13,3,4,6,12,17,21,8,6,7,16,9,2,
                            12,8,5,6,8,7,14,15,9,7 ,#Julio 2021
                            
                            11,17,3,7,8,10,18,13,11,12,9,6,3,7,21,9,9,14,15,3,13,13,13,7,6,5,12,6,16,14,13,14,
                               10,2,2,15,12,11,5,5,9,8,13,12,13,15,7,8,17,21,20,4,13,7,7,7,9,9,10,12,2,8, #Agosto 2021 B
                               
                            16,11,7,16,11,3,7,7,12,3,13,14,9,6,13,4,2,6,15,10,12,6,8,5,13,
                               12,22,7,11,5,12,8,16,8,2,9,11,6,10,8,5,2,7,4,15,14,8,12,10,9,
                               17,19,21,17,8,7,7,16,7,8,#SEPTIEMBRE B
                            
                            16,5,7,10,4,6,10,18,5,8,15,9,18,3,13,10,10,10,18,9,14,9,7,11,
                               4,6,12,10,11,8,10,16,10,13,9,2,15,17,11,14,5,11,10,9,8,5,
                               7,3,3,8,5,9,11,14,4,6,8,6,2,15,3,5, #OCTOBER B
                            
                            14,4,11,15,12,12,3,11,9,8,5,6,17,12,6,4,8,10,4,8,2,9,7,18,9,9,16,6,
                               4,14,15,9,9,17,7,6,16,6,14,19,13,19,10,11,12,3,4,9,14,8,9,7,14,9,
                               2,3,15,12,8,6, #November b
                               
                            12,9,5,6,6,13,8,7,8,10,3,14,17,7,6,21,13,17,16,10,18,12,5,7,7,16,
                            7,14,9,6,11,7,15,4,8,2,15,10,5,4,6,7,13,2,5,9,2,12,4,9,2,12,2,14,
                               4,4,6,14,5,17,8,2, #dICIEMBRE B
                               
                               #ener0 b 2022
                               10,3,12,9,14,11,8,18,10,6,5,5,18,14,15,6,8,9,11,9],
                                 
                                 index=['Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                     
                                    'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                  
                                    'Enero 31','Enero 31 ','Enero 30','Enero 30 ','Enero 29','Enero 29 ',
                                    'Enero 28','Enero 28 ','Enero 27','Enero 27 ',
                                    'Enero 26','Enero 26 ','Enero 25','Enero 25 ',
                                    'Enero 24','Enero 24 ','Enero 23','Enero 23 ',
                                    'Enero 22','Enero 22 ','Enero 21','Enero 21 ',
                                    'Enero 20','Enero 20 ','Enero 19','Enero 19 ',
                                    'Enero 18','Enero 18 ','Enero 17','Enero 17 ',
                                    'Enero 16','Enero 16 ','Enero 15','Enero 15 ',
                                    'Enero 14','Enero 14 ','Enero 13','Enero 13 ',
                                    'Enero 12','Enero 12 ','Enero 11','Enero 11 ',
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ',
                                  
                                    'Febrero  28','Febrero  28 ','Febrero  27','Febrero  27 ',
                                    'Febrero  26','Febrero  26 ','Febrero  25','Febrero  25 ',
                                    'Febrero  24','Febrero  24 ','Febrero  23','Febrero  23 ',
                                    'Febrero  22','Febrero  22 ','Febrero  21','Febrero  21 ',
                                    'Febrero  20','Febrero  20 ','Febrero  19','Febrero  19 ',
                                    'Febrero  18','Febrero  18 ','Febrero  17','Febrero  17 ',
                                    'Febrero  16','Febrero  16 ','Febrero  15','Febrero  15 ',
                                    'Febrero  14','Febrero  14 ','Febrero  13','Febrero  13 ',
                                    'Febrero  12','Febrero  12 ','Febrero  11','Febrero  11 ',
                                    'Febrero  10','Febrero  10 ','Febrero  9 ','Febrero  9 ',
                                    'Febrero  8 ','Febrero  8  ','Febrero  7 ','Febrero  7 ',
                                    'Febrero  6 ','Febrero  6  ','Febrero  5 ','Febrero  5 ',
                                    'Febrero  4 ','Febrero  4  ','Febrero  3 ','Febrero  3 ',
                                    'Febrero  2 ','Febrero  2  ','Febrero  1 ','Febrero  1 ',
                                   
                                   'Marzo  31','Marzo  31 ', 'Marzo  30',
                                    'Marzo  30 ','Marzo  29','Marzo  29 ',
                                    'Marzo  28','Marzo  28 ','Marzo  27','Marzo  27 ',
                                    'Marzo  26','Marzo  26 ','Marzo  25','Marzo  25 ',
                                    'Marzo  24','Marzo  24 ','Marzo  23','Marzo  23 ',
                                    'Marzo  22','Marzo  22 ','Marzo  21','Marzo  21 ',
                                    'Marzo  20','Marzo  20 ','Marzo  19','Marzo  19 ',
                                    'Marzo  18','Marzo  18 ','Marzo  17','Marzo  17 ',
                                    'Marzo  16','Marzo  16 ','Marzo  15','Marzo  15 ',
                                    'Marzo  14','Marzo  14 ','Marzo  13','Marzo  13 ',
                                    'Marzo  12','Marzo  12 ','Marzo  11','Marzo  11 ',
                                    'Marzo  10','Marzo  10 ','Marzo  9 ','Marzo  9 ',
                                    'Marzo  8 ','Marzo  8  ','Marzo  7 ','Marzo  7 ',
                                    'Marzo  6 ','Marzo  6  ','Marzo  5 ','Marzo  5 ',
                                    'Marzo  4 ','Marzo  4  ','Marzo  3 ','Marzo  3 ',
                                    'Marzo  2 ','Marzo  2  ','Marzo  1 ','Marzo  1 ',
                                   
                                   'Abril  30','Abril  30 ','Abril  29','Abril  29 ',
                                    'Abril  28','Abril  28 ','Abril  27','Abril  27 ',
                                    'Abril  26','Abril  26 ','Abril  25','Abril  25 ',
                                    'Abril  24','Abril  24 ','Abril  23','Abril  23 ',
                                    'Abril  22','Abril  22 ','Abril  21','Abril  21 ',
                                    'Abril  20','Abril  20 ','Abril  19','Abril  19 ',
                                    'Abril  18','Abril  18 ','Abril  17','Abril  17 ',
                                    'Abril  16','Abril  16 ','Abril  15','Abril  15 ',
                                    'Abril  14','Abril  14 ','Abril  13','Abril  13 ',
                                    'Abril  12','Abril  12 ','Abril  11','Abril  11 ',
                                    'Abril  10','Abril  10 ','Abril  9 ','Abril  9 ',
                                    'Abril  8 ','Abril  8  ','Abril  7 ','Abril  7 ',
                                    'Abril  6 ','Abril  6  ','Abril  5 ','Abril  5 ',
                                    'Abril  4 ','Abril  4  ','Abril  3 ','Abril  3 ',
                                    'Abril  2 ','Abril  2  ','Abril  1 ','Abril  1 ',
                                   
                                   'Mayo  31','Mayo  31 ',
                                    'Mayo  30','Mayo  30 ','Mayo  29','Mayo  29 ',
                                    'Mayo  28','Mayo  28 ','Mayo  27','Mayo  27 ',
                                    'Mayo  26','Mayo  26 ','Mayo  25','Mayo  25 ',
                                    'Mayo  24','Mayo  24 ','Mayo  23','Mayo  23 ',
                                    'Mayo  22','Mayo  22 ','Mayo  21','Mayo  21 ',
                                    'Mayo  20','Mayo  20 ','Mayo  19','Mayo  19 ',
                                    'Mayo  18','Mayo  18 ','Mayo  17','Mayo  17 ',
                                    'Mayo  16','Mayo  16 ','Mayo  15','Mayo  15 ',
                                    'Mayo  14','Mayo  14 ','Mayo  13','Mayo  13 ',
                                    'Mayo  12','Mayo  12 ','Mayo  11','Mayo  11 ',
                                    'Mayo  10','Mayo  10 ','Mayo  9 ','Mayo  9 ',
                                    'Mayo  8 ','Mayo  8  ','Mayo  7 ','Mayo  7 ',
                                    'Mayo  6 ','Mayo  6  ','Mayo  5 ','Mayo  5 ',
                                    'Mayo  4 ','Mayo  4  ','Mayo  3 ','Mayo  3 ',
                                    'Mayo  2 ','Mayo  2  ','Mayo  1 ','Mayo  1 ',
                                   
                                    'Junio 30','Junio 30 ','Junio 29','Junio 29 ',
                                    'Junio 28','Junio 28 ','Junio 27','Junio 27 ',
                                    'Junio 26','Junio 26 ','Junio 25','Junio 25 ',
                                    'Junio 24','Junio 24 ','Junio 23','Junio 23 ',
                                    'Junio 22','Junio 22 ','Junio 21','Junio 21 ',
                                    'Junio 20','Junio 20 ','Junio 19','Junio 19 ',
                                    'Junio 18','Junio 18 ','Junio 17','Junio 17 ',
                                    'Junio 16','Junio 16 ','Junio 15','Junio 15 ',
                                    'Junio 14','Junio 14 ','Junio 13','Junio 13 ',
                                    'Junio 12','Junio 12 ','Junio 11','Junio 11 ',
                                    'Junio 10','Junio 10 ','Junio 9 ','Junio 9 ',
                                    'Junio 8 ','Junio 8  ','Junio 7 ','Junio 7 ',
                                    'Junio 6 ','Junio 6  ','Junio 5 ','Junio 5 ',
                                    'Junio 4 ','Junio 4  ','Junio 3 ','Junio 3 ',
                                    'Junio 2 ','Junio 2  ','Junio 1 ','Junio 1 ',
                                   
                                   'Julio 31','Julio 31','Julio 30','Julio 30 ','Julio 29','Julio 29 ',
                                    'Julio 28','Julio 28 ','Julio 27','Julio 27 ',
                                    'Julio 26','Julio 26 ','Julio 25','Julio 25 ',
                                    'Julio 24','Julio 24 ','Julio 23','Julio 23 ',
                                    'Julio 22','Julio 22 ','Julio 21','Julio 21 ',
                                    'Julio 20','Julio 20 ','Julio 19','Julio 19 ',
                                    'Julio 18','Julio 18 ','Julio 17','Julio 17 ',
                                    'Julio 16','Julio 16 ','Julio 15','Julio 15 ',
                                    'Julio 14','Julio 14 ','Julio 13','Julio 13 ',
                                    'Julio 12','Julio 12 ','Julio 11','Julio 11 ',
                                    'Julio 10','Julio 10 ','Julio 9 ','Julio 9 ',
                                    'Julio 8 ','Julio 8  ','Julio 7 ','Julio 7 ',
                                    'Julio 6 ','Julio 6  ','Julio 5 ','Julio 5 ',
                                    'Julio 4 ','Julio 4  ','Julio 3 ','Julio 3 ',
                                    'Julio 2 ','Julio 2  ','Julio 1 ','Julio 1 ',
                                   
                                   'Agosto 31','Agosto 31','Agosto 30','Agosto 30 ','Agosto 29','Agosto 29 ',
                                    'Agosto 28','Agosto 28 ','Agosto 27','Agosto 27 ',
                                    'Agosto 26','Agosto 26 ','Agosto 25','Agosto 25 ',
                                    'Agosto 24','Agosto 24 ','Agosto 23','Agosto 23 ',
                                    'Agosto 22','Agosto 22 ','Agosto 21','Agosto 21 ',
                                    'Agosto 20','Agosto 20 ','Agosto 19','Agosto 19 ',
                                    'Agosto 18','Agosto 18 ','Agosto 17','Agosto 17 ',
                                    'Agosto 16','Agosto 16 ','Agosto 15','Agosto 15 ',
                                    'Agosto 14','Agosto 14 ','Agosto 13','Agosto 13 ',
                                    'Agosto 12','Agosto 12 ','Agosto 11','Agosto 11 ',
                                    'Agosto 10','Agosto 10 ','Agosto 9 ','Agosto 9 ',
                                    'Agosto 8 ','Agosto 8  ','Agosto 7 ','Agosto 7 ',
                                    'Agosto 6 ','Agosto 6  ','Agosto 5 ','Agosto 5 ',
                                    'Agosto 4 ','Agosto 4  ','Agosto 3 ','Agosto 3 ',
                                    'Agosto 2 ','Agosto 2  ','Agosto 1 ','Agosto 1 ',
                                   
                                   'Septiembre 30','Septiembre 30 ','Septiembre 29','Septiembre 29 ',
                                    'Septiembre 28','Septiembre 28 ','Septiembre 27','Septiembre 27 ',
                                    'Septiembre 26','Septiembre 26 ','Septiembre 25','Septiembre 25 ',
                                    'Septiembre 24','Septiembre 24 ','Septiembre 23','Septiembre 23 ',
                                    'Septiembre 22','Septiembre 22 ','Septiembre 21','Septiembre 21 ',
                                    'Septiembre 20','Septiembre 20 ','Septiembre 19','Septiembre 19 ',
                                    'Septiembre 18','Septiembre 18 ','Septiembre 17','Septiembre 17 ',
                                    'Septiembre 16','Septiembre 16 ','Septiembre 15','Septiembre 15 ',
                                    'Septiembre 14','Septiembre 14 ','Septiembre 13','Septiembre 13 ',
                                    'Septiembre 12','Septiembre 12 ','Septiembre 11','Septiembre 11 ',
                                    'Septiembre 10','Septiembre 10 ','Septiembre 9 ','Septiembre 9 ',
                                    'Septiembre 8 ','Septiembre 8  ','Septiembre 7 ','Septiembre 7 ',
                                    'Septiembre 6 ','Septiembre 6  ','Septiembre 5 ','Septiembre 5 ',
                                    'Septiembre 4 ','Septiembre 4  ','Septiembre 3 ','Septiembre 3 ',
                                    'Septiembre 2 ','Septiembre 2  ','Septiembre 1 ','Septiembre 1 ',
                                   
                                   'Octubre 31','Octubre 31','Octubre 30','Octubre 30 ','Octubre 29','Octubre 29 ',
                                    'Octubre 28','Octubre 28 ','Octubre 27','Octubre 27 ',
                                    'Octubre 26','Octubre 26 ','Octubre 25','Octubre 25 ',
                                    'Octubre 24','Octubre 24 ','Octubre 23','Octubre 23 ',
                                    'Octubre 22','Octubre 22 ','Octubre 21','Octubre 21 ',
                                    'Octubre 20','Octubre 20 ','Octubre 19','Octubre 19 ',
                                    'Octubre 18','Octubre 18 ','Octubre 17','Octubre 17 ',
                                    'Octubre 16','Octubre 16 ','Octubre 15','Octubre 15 ',
                                    'Octubre 14','Octubre 14 ','Octubre 13','Octubre 13 ',
                                    'Octubre 12','Octubre 12 ','Octubre 11','Octubre 11 ',
                                    'Octubre 10','Octubre 10 ','Octubre 9 ','Octubre 9 ',
                                    'Octubre 8 ','Octubre 8  ','Octubre 7 ','Octubre 7 ',
                                    'Octubre 6 ','Octubre 6  ','Octubre 5 ','Octubre 5 ',
                                    'Octubre 4 ','Octubre 4  ','Octubre 3 ','Octubre 3 ',
                                    'Octubre 2 ','Octubre 2  ','Octubre 1 ','Octubre 1 ',
                                   
                                   'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                   
                                   'Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                    
                                   'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ' ]),
           'Valor C' :pd.Series ([12,12,21,15,22,16,22,25,22,14,18,11,10,10,16,19,12,21,19,19,21,16,19,16,19,
                             18,12,15,21,15,12,13,13,15,10,16,11,12,6,9,7,21,13,19,12,14,22,10,20,
                             19,10,19,14,10,12,22,11,8,14,16,5,19,#diciembre 2020
                             
                             9,15,14,24,12,5,10,15,6,13,14,12,16,13,20,8,8,14,10,13,6,10,
                             16,17,11,13,5,12,3,9,10,10,24,22,13,12,22,17,14,18,20,18,23,
                             20,15,13,10,15,7,11,12,6,14,10,19,3,5,20,7,14, #Noviembre 2020
                             
                             17,13,22,12,11,14,23,10,10,15,19,13,18,15,11,13,11,13,10,11,24,11,22,14,12,6,8,14,18,17,
                             14,15,13,17,12,12,11,15,12,8,18,12,16,12,15,10,14,16,11,11,14,22,
                             9,15,10,14,17,9,8,6,17,18,#Enero2021
                             
                             23,13,17,11,6,14,16,19,10,13,13,9,17,15,16,14,21,10,6,16,19,20,10,24,11,
                             18,16,13,16,12,20,14,14,14,14,18,6,23,16,20,13,17,11,7,9,15,25,14,7,11,7,
                             21,20,6,15,16, #Febrero 2021
                            
                             12,19,13,24,22,10,18,10,22,5,8,12,17,17,10,19,19,8,18,21,4,15,21,
                             15,5,14,19,20,20,8,12,8,23,9,11,6,15,20,18,20,10,10,17,9,9,10,12,23,
                             9,17,18,20,20,19,21,12,7,12,9,20,21,20,   #Marzo2021
                               
                             17,11,12,13,10,18,12,10,12,13,17,10,13,12,18,12,8,13,18,15,
                             14,15,22,11,8,17,24,6,10,8,18,13,21,16,14,15,7,15,11,12,6,25,
                             6,13,9,23,11,5,11,7,14,14,9,22,14,14,9,5,18,23, #Abril 2021
                            
                             8,12,21,14,14,13,7,17,11,14,20,14,13,23,17,24,13,16,
                             9,13,7,7,3,19,12,22,14,13,13,12,11,9,20,8,13,16,11,13,17,15,13,10,18,
                             17,16,12,22,17,15,18,13,5,11,6,9,24,15,11,15,23,17,13, #MAYO 2021
                               
                             20,16,3,10,7,17,3,9,12,16,18,8,16,21,13,23,9,16,19,16,
                               21,9,7,19,14,16,9,13,11,14,21,12,11,12,9,12,21,21,10,18,19,13,
                               10,11,11,23,15,12,23,14,20,6,19,7,9,17,13,20,12,22, #Junio 2021
                               
                             20,15,19,23,7,10,8,22,21,23,21,13,9,8,15,15,11,13,118,12,16,16,18,16,16,12,
                               16,8,14,16,11,13,16,16,21,8,9,10,10,16,6,17,7,15,19,22,22,8,11,23,13,10,
                               17,10,7,8,10,9,17,19,10,12, #Julio 2021 c
                               
                             12,22,5,8,10,11,24,19,16,19,21,15,4,10,24,15,12,14,23,4,17,19,16,17,20,15,
                               16,18,18,20,21,19,13,6,9,20,17,16,19,8,15,12,18,17,14,16,10,9,18,22,21,10,19,
                               11,13,10,12,10,14,15,13,13, #Agosto 2021 c
                             
                             22,12,12,18,17,9,11,15,15,8,14,18,20,8,20,6,3,15,18,16,15,
                               18,16,11,16,21,24,14,16,18,13,17,18,16,4,17,17,10,14,20,6,10,
                               8,6,18,22,22,16,12,13,21,23,22,17,14,10,10,17,11,12,#SEPTIEMBRE C
                               
                             18,8,8,13,10,10,13,19,10,14,16,12,23,21,20,16,14,11,22,17,16,17,10,16,
                               5,13,17,12,15,18,14,17,16,21,21,3,17,23,22,15,13,12,14,12,12,14,
                               14,12,5,16,7,12,16,16,7,13,10,12,7,18,16,8, #OCTOBER C
                               
                             16,8,15,18,14,14,12,21,16,9,18,17,19,18,8,6,19,15,20,18,13,19,16,21,10,15,
                              18,10,18,15,19,14,22,19,8,15,17,9,23,22,17,21,13,13,14,17,6,14,18,10,
                               10,17,19,19,5,14,18,14,17,7, #November c
                               
                             13,15,10,9,10,21,13,11,12,18,10,14,18,144,15,22,14,19,24,19,20,18,
                             13,10,15,19,15,19,12,13,9,17,22,13,21,9,23,19,9,12,12,9,15,8,13,10,20,23,
                               16,11,10,19,5,15,14,16,9,20,17,18,9,3, #dICIEMBREC
                               
                               #enero c 2022
                               12,11,17,11,16,12,13,19,17,11,8,11,23,15,22,8,13,20,16,17
                               
                              ],
                             index=['Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                   
                                   'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                   
                                    'Enero 31','Enero 31 ','Enero 30','Enero 30 ','Enero 29','Enero 29 ',
                                    'Enero 28','Enero 28 ','Enero 27','Enero 27 ',
                                    'Enero 26','Enero 26 ','Enero 25','Enero 25 ',
                                    'Enero 24','Enero 24 ','Enero 23','Enero 23 ',
                                    'Enero 22','Enero 22 ','Enero 21','Enero 21 ',
                                    'Enero 20','Enero 20 ','Enero 19','Enero 19 ',
                                    'Enero 18','Enero 18 ','Enero 17','Enero 17 ',
                                    'Enero 16','Enero 16 ','Enero 15','Enero 15 ',
                                    'Enero 14','Enero 14 ','Enero 13','Enero 13 ',
                                    'Enero 12','Enero 12 ','Enero 11','Enero 11 ',
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ',
                                    
                                    'Febrero  28','Febrero  28 ','Febrero  27','Febrero  27 ',
                                    'Febrero  26','Febrero  26 ','Febrero  25','Febrero  25 ',
                                    'Febrero  24','Febrero  24 ','Febrero  23','Febrero  23 ',
                                    'Febrero  22','Febrero  22 ','Febrero  21','Febrero  21 ',
                                    'Febrero  20','Febrero  20 ','Febrero  19','Febrero  19 ',
                                    'Febrero  18','Febrero  18 ','Febrero  17','Febrero  17 ',
                                    'Febrero  16','Febrero  16 ','Febrero  15','Febrero  15 ',
                                    'Febrero  14','Febrero  14 ','Febrero  13','Febrero  13 ',
                                    'Febrero  12','Febrero  12 ','Febrero  11','Febrero  11 ',
                                    'Febrero  10','Febrero  10 ','Febrero  9 ','Febrero  9 ',
                                    'Febrero  8 ','Febrero  8  ','Febrero  7 ','Febrero  7 ',
                                    'Febrero  6 ','Febrero  6  ','Febrero  5 ','Febrero  5 ',
                                    'Febrero  4 ','Febrero  4  ','Febrero  3 ','Febrero  3 ',
                                    'Febrero  2 ','Febrero  2  ','Febrero  1 ','Febrero  1 ',
                                    
                                    'Marzo  31','Marzo  31 ', 'Marzo  30',
                                    'Marzo  30 ','Marzo  29','Marzo  29 ',
                                    'Marzo  28','Marzo  28 ','Marzo  27','Marzo  27 ',
                                    'Marzo  26','Marzo  26 ','Marzo  25','Marzo  25 ',
                                    'Marzo  24','Marzo  24 ','Marzo  23','Marzo  23 ',
                                    'Marzo  22','Marzo  22 ','Marzo  21','Marzo  21 ',
                                    'Marzo  20','Marzo  20 ','Marzo  19','Marzo  19 ',
                                    'Marzo  18','Marzo  18 ','Marzo  17','Marzo  17 ',
                                    'Marzo  16','Marzo  16 ','Marzo  15','Marzo  15 ',
                                    'Marzo  14','Marzo  14 ','Marzo  13','Marzo  13 ',
                                    'Marzo  12','Marzo  12 ','Marzo  11','Marzo  11 ',
                                    'Marzo  10','Marzo  10 ','Marzo  9 ','Marzo  9 ',
                                    'Marzo  8 ','Marzo  8  ','Marzo  7 ','Marzo  7 ',
                                    'Marzo  6 ','Marzo  6  ','Marzo  5 ','Marzo  5 ',
                                    'Marzo  4 ','Marzo  4  ','Marzo  3 ','Marzo  3 ',
                                    'Marzo  2 ','Marzo  2  ','Marzo  1 ','Marzo  1 ',
                                    
                                    'Abril  30','Abril  30 ','Abril  29','Abril  29 ',
                                    'Abril  28','Abril  28 ','Abril  27','Abril  27 ',
                                    'Abril  26','Abril  26 ','Abril  25','Abril  25 ',
                                    'Abril  24','Abril  24 ','Abril  23','Abril  23 ',
                                    'Abril  22','Abril  22 ','Abril  21','Abril  21 ',
                                    'Abril  20','Abril  20 ','Abril  19','Abril  19 ',
                                    'Abril  18','Abril  18 ','Abril  17','Abril  17 ',
                                    'Abril  16','Abril  16 ','Abril  15','Abril  15 ',
                                    'Abril  14','Abril  14 ','Abril  13','Abril  13 ',
                                    'Abril  12','Abril  12 ','Abril  11','Abril  11 ',
                                    'Abril  10','Abril  10 ','Abril  9 ','Abril  9 ',
                                    'Abril  8 ','Abril  8  ','Abril  7 ','Abril  7 ',
                                    'Abril  6 ','Abril  6  ','Abril  5 ','Abril  5 ',
                                    'Abril  4 ','Abril  4  ','Abril  3 ','Abril  3 ',
                                    'Abril  2 ','Abril  2  ','Abril  1 ','Abril  1 ',
                                    
                                    'Mayo  31','Mayo  31 ',
                                    'Mayo  30','Mayo  30 ','Mayo  29','Mayo  29 ',
                                    'Mayo  28','Mayo  28 ','Mayo  27','Mayo  27 ',
                                    'Mayo  26','Mayo  26 ','Mayo  25','Mayo  25 ',
                                    'Mayo  24','Mayo  24 ','Mayo  23','Mayo  23 ',
                                    'Mayo  22','Mayo  22 ','Mayo  21','Mayo  21 ',
                                    'Mayo  20','Mayo  20 ','Mayo  19','Mayo  19 ',
                                    'Mayo  18','Mayo  18 ','Mayo  17','Mayo  17 ',
                                    'Mayo  16','Mayo  16 ','Mayo  15','Mayo  15 ',
                                    'Mayo  14','Mayo  14 ','Mayo  13','Mayo  13 ',
                                    'Mayo  12','Mayo  12 ','Mayo  11','Mayo  11 ',
                                    'Mayo  10','Mayo  10 ','Mayo  9 ','Mayo  9 ',
                                    'Mayo  8 ','Mayo  8  ','Mayo  7 ','Mayo  7 ',
                                    'Mayo  6 ','Mayo  6  ','Mayo  5 ','Mayo  5 ',
                                    'Mayo  4 ','Mayo  4  ','Mayo  3 ','Mayo  3 ',
                                    'Mayo  2 ','Mayo  2  ','Mayo  1 ','Mayo  1 ',
                                    
                                    'Junio 30','Junio 30 ','Junio 29','Junio 29 ',
                                    'Junio 28','Junio 28 ','Junio 27','Junio 27 ',
                                    'Junio 26','Junio 26 ','Junio 25','Junio 25 ',
                                    'Junio 24','Junio 24 ','Junio 23','Junio 23 ',
                                    'Junio 22','Junio 22 ','Junio 21','Junio 21 ',
                                    'Junio 20','Junio 20 ','Junio 19','Junio 19 ',
                                    'Junio 18','Junio 18 ','Junio 17','Junio 17 ',
                                    'Junio 16','Junio 16 ','Junio 15','Junio 15 ',
                                    'Junio 14','Junio 14 ','Junio 13','Junio 13 ',
                                    'Junio 12','Junio 12 ','Junio 11','Junio 11 ',
                                    'Junio 10','Junio 10 ','Junio 9 ','Junio 9 ',
                                    'Junio 8 ','Junio 8  ','Junio 7 ','Junio 7 ',
                                    'Junio 6 ','Junio 6  ','Junio 5 ','Junio 5 ',
                                    'Junio 4 ','Junio 4  ','Junio 3 ','Junio 3 ',
                                    'Junio 2 ','Junio 2  ','Junio 1 ','Junio 1 ',
                                    
                                    'Julio 31','Julio 31','Julio 30','Julio 30 ','Julio 29','Julio 29 ',
                                    'Julio 28','Julio 28 ','Julio 27','Julio 27 ',
                                    'Julio 26','Julio 26 ','Julio 25','Julio 25 ',
                                    'Julio 24','Julio 24 ','Julio 23','Julio 23 ',
                                    'Julio 22','Julio 22 ','Julio 21','Julio 21 ',
                                    'Julio 20','Julio 20 ','Julio 19','Julio 19 ',
                                    'Julio 18','Julio 18 ','Julio 17','Julio 17 ',
                                    'Julio 16','Julio 16 ','Julio 15','Julio 15 ',
                                    'Julio 14','Julio 14 ','Julio 13','Julio 13 ',
                                    'Julio 12','Julio 12 ','Julio 11','Julio 11 ',
                                    'Julio 10','Julio 10 ','Julio 9 ','Julio 9 ',
                                    'Julio 8 ','Julio 8  ','Julio 7 ','Julio 7 ',
                                    'Julio 6 ','Julio 6  ','Julio 5 ','Julio 5 ',
                                    'Julio 4 ','Julio 4  ','Julio 3 ','Julio 3 ',
                                    'Julio 2 ','Julio 2  ','Julio 1 ','Julio 1 ',
                                    
                                    'Agosto 31','Agosto 31','Agosto 30','Agosto 30 ','Agosto 29','Agosto 29 ',
                                    'Agosto 28','Agosto 28 ','Agosto 27','Agosto 27 ',
                                    'Agosto 26','Agosto 26 ','Agosto 25','Agosto 25 ',
                                    'Agosto 24','Agosto 24 ','Agosto 23','Agosto 23 ',
                                    'Agosto 22','Agosto 22 ','Agosto 21','Agosto 21 ',
                                    'Agosto 20','Agosto 20 ','Agosto 19','Agosto 19 ',
                                    'Agosto 18','Agosto 18 ','Agosto 17','Agosto 17 ',
                                    'Agosto 16','Agosto 16 ','Agosto 15','Agosto 15 ',
                                    'Agosto 14','Agosto 14 ','Agosto 13','Agosto 13 ',
                                    'Agosto 12','Agosto 12 ','Agosto 11','Agosto 11 ',
                                    'Agosto 10','Agosto 10 ','Agosto 9 ','Agosto 9 ',
                                    'Agosto 8 ','Agosto 8  ','Agosto 7 ','Agosto 7 ',
                                    'Agosto 6 ','Agosto 6  ','Agosto 5 ','Agosto 5 ',
                                    'Agosto 4 ','Agosto 4  ','Agosto 3 ','Agosto 3 ',
                                    'Agosto 2 ','Agosto 2  ','Agosto 1 ','Agosto 1 ',
                                    
                                    'Septiembre 30','Septiembre 30 ','Septiembre 29','Septiembre 29 ',
                                    'Septiembre 28','Septiembre 28 ','Septiembre 27','Septiembre 27 ',
                                    'Septiembre 26','Septiembre 26 ','Septiembre 25','Septiembre 25 ',
                                    'Septiembre 24','Septiembre 24 ','Septiembre 23','Septiembre 23 ',
                                    'Septiembre 22','Septiembre 22 ','Septiembre 21','Septiembre 21 ',
                                    'Septiembre 20','Septiembre 20 ','Septiembre 19','Septiembre 19 ',
                                    'Septiembre 18','Septiembre 18 ','Septiembre 17','Septiembre 17 ',
                                    'Septiembre 16','Septiembre 16 ','Septiembre 15','Septiembre 15 ',
                                    'Septiembre 14','Septiembre 14 ','Septiembre 13','Septiembre 13 ',
                                    'Septiembre 12','Septiembre 12 ','Septiembre 11','Septiembre 11 ',
                                    'Septiembre 10','Septiembre 10 ','Septiembre 9 ','Septiembre 9 ',
                                    'Septiembre 8 ','Septiembre 8  ','Septiembre 7 ','Septiembre 7 ',
                                    'Septiembre 6 ','Septiembre 6  ','Septiembre 5 ','Septiembre 5 ',
                                    'Septiembre 4 ','Septiembre 4  ','Septiembre 3 ','Septiembre 3 ',
                                    'Septiembre 2 ','Septiembre 2  ','Septiembre 1 ','Septiembre 1 ',
                                    
                                    'Octubre 31','Octubre 31','Octubre 30','Octubre 30 ','Octubre 29','Octubre 29 ',
                                    'Octubre 28','Octubre 28 ','Octubre 27','Octubre 27 ',
                                    'Octubre 26','Octubre 26 ','Octubre 25','Octubre 25 ',
                                    'Octubre 24','Octubre 24 ','Octubre 23','Octubre 23 ',
                                    'Octubre 22','Octubre 22 ','Octubre 21','Octubre 21 ',
                                    'Octubre 20','Octubre 20 ','Octubre 19','Octubre 19 ',
                                    'Octubre 18','Octubre 18 ','Octubre 17','Octubre 17 ',
                                    'Octubre 16','Octubre 16 ','Octubre 15','Octubre 15 ',
                                    'Octubre 14','Octubre 14 ','Octubre 13','Octubre 13 ',
                                    'Octubre 12','Octubre 12 ','Octubre 11','Octubre 11 ',
                                    'Octubre 10','Octubre 10 ','Octubre 9 ','Octubre 9 ',
                                    'Octubre 8 ','Octubre 8  ','Octubre 7 ','Octubre 7 ',
                                    'Octubre 6 ','Octubre 6  ','Octubre 5 ','Octubre 5 ',
                                    'Octubre 4 ','Octubre 4  ','Octubre 3 ','Octubre 3 ',
                                    'Octubre 2 ','Octubre 2  ','Octubre 1 ','Octubre 1 ',
                                    
                                    'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                    
                                    'Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                    
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 '
                                   ]),
       
       'Valor D' :pd.Series ([17,15,25,21,26,26,25,26,24,20,20,19,13,17,18,20,20,23,20,21,27,21,24,19,
                            23,23,20,24,24,24,23,15,14,17,12,24,12,23,11,13,9,23,16,22,
                            17,21,14,18,24,24,17,24,25,12,14,24,14,9,16,26,6,21,#Diciembre 2020
                            
                            12,22,17,27,17,7,11,25,21,19,17,14,21,16,24,9,20,17,18,22,21,21,
                            24,26,22,18,12,18,21,12,12,17,26,24,20,14,23,19,15,25,26,19,24,
                            21,23,14,17,22,22,12,15,10,22,23,22,8,7,23,10,18,#Noviembre2020
                             
                            21,18,25,15,17,22,25,11,23,16,24,19,
                            22,21,12,18,12,14,21,14,26,13,23,14,
                            23,13,14,16,20,18,15,18,13,21,13,21,23,
                            17,14,21,19,21,18,19,25,16,17,21,25,12,17,23,
                            14,16,11,26,20,14,13,10,18,19,#Enero 2021
                             
                            26,24,20,18,14,15,17,20,21,21,20,25,19,17,21,16,22,13,9,17,21,27,
                            14,26,26,21,19,24,17,22,25,18,20,21,24,19,10,24,25,26,21,
                            19,17,25,19,24,27,15,11,22,10,22,26,15,28,19, #Febrero 2021
                            
                            16,20,16,26,23,12,22,17,23,21,15,19,20,19,11,21,22,15,22,24,6,22,24,26,6,19,
                            21,25,22,14,24,21,25,11,20,11,20,21,26,23,12,16,20,15,16,11,23,25,11,20,21,24,
                            23,26,22,18,11,16,19,21,22,22, #Marzo
                              
                            19,18,18,17,12,19,22,18,13,15,25,13,19,19,22,20,18,22,23,
                            27,16,19,25,25,11,20,24,12,23,13,24,18,25,20,18,18,24,16,15,17,7,27,
                            19,16,10,27,24,11,16,12,17,16,12,25,16,18,22,20,24,26, #Abril
                             
                            13,18,22,19,22,14,17,18,12,15,26,17,20,24,21,25,21,17,18,22,10,12,
                            4,24,20,23,18,14,19,22,22,12,26,15,14,23,15,20,25,17,16,13,23,22,21,
                            22,26,21,20,21,18,14,14,10,16,24,19,21,23,27,21,14, #MAYO 2021
                              
                            22,17,16,22,17,18,11,10,15,17,20,17,20,25,15,24,22,23,
                            21,22,24,18,18,24,19,23,21,19,17,21,25,14,17,24,10,15,22,23,
                            15,19,22,14,15,16,20,25,21,12,24,18,21,17,20,14,11,20,25,24,19,25, #JUNIO 2021
                              
                            26,16,22,25,14,14,12,25,22,26,22,16,10,9,18,17,14,23,23,20,20,20,23,22,
                              18,16,21,12,21,18,12,18,19,18,24,17,17,12,20,224,10,24,16,20,23,27,24,9,13,
                              25,18,12,18,14,15,10,20,18,22,26,16,19, #Julio 2021
                             
                             24,25,88,14,16,13,27,20,19,23,26,17,5,11,26,16,18,16,24,10,24,26,18,20,24,17,
                              21,21,27,24,23,25,25,16,12,23,19,24,23,11,17,16,25,18,17,18,18,18,23,25,26,13,20,22,
                              17,16,19,17,16,19,25,16, #Agosto D
                              
                             23,20,16,19,26,23,20,16,18,12,27,23,21,14,21,7,16,21,19,17,22,24,18,12,22,26,
                              25,17,22,21,19,21,21,17,14,20,19,18,17,24,7,17,14,12,19,26,24,21,17,18,23,27,27,26,
                              17,14,17,20,19,19,#SEPRIEMBRE D
                              
                             24,14,12,14,16,12,17,22,25,16,17,25,27,25,21,17,19,12,23,18,21,24,12,19,
                             16,21,19,15,26,21,18,20,20,26,23,5,19,26,27,19,14,13,25,
                              13,18,19,27,21,6,17,22,15,22,20,14,16,22,22,26,20,20,12, #OCTOBER D
                              
                              
                            18,25,21,25,22,22,12,22,17,26,19,23,21,21,13,10,20,18,22,23,17,23,25,
                              24,19,22,28,15,19,23,22,18,25,22,11,18,17,10,24,23,22,22,19,14,19,19,22,19,24,
                              20,20,18,22,20,8,15,22,15,22,17, #NOVIEMBRE D
                              
                            17,19,15,19,11,25,22,13,14,19,20,24,20,20,18,24,15,21,25,20,24,24,23,16,17,22,
                            19,20,14,16,25,18,23,24,24,15,24,20,16,26,19,13,16,11,16,16,26,24,20,16,11,26,
                              12,17,17,18,10,22,21,20,10,7, #Diciembre d
                              
                              #enero d 2022
                              17,21,22,16,21,15,16,23,24,12,9,17,24,21,25,9,17,21,18,21
                              
                             ],
                            index=['Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                   
                                   'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                  
                                     'Enero 31','Enero 31 ','Enero 30','Enero 30 ','Enero 29','Enero 29 ',
                                    'Enero 28','Enero 28 ','Enero 27','Enero 27 ',
                                    'Enero 26','Enero 26 ','Enero 25','Enero 25 ',
                                    'Enero 24','Enero 24 ','Enero 23','Enero 23 ',
                                    'Enero 22','Enero 22 ','Enero 21','Enero 21 ',
                                    'Enero 20','Enero 20 ','Enero 19','Enero 19 ',
                                    'Enero 18','Enero 18 ','Enero 17','Enero 17 ',
                                    'Enero 16','Enero 16 ','Enero 15','Enero 15 ',
                                    'Enero 14','Enero 14 ','Enero 13','Enero 13 ',
                                    'Enero 12','Enero 12 ','Enero 11','Enero 11 ',
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ',
                                   
                                   'Febrero  28','Febrero  28 ','Febrero  27','Febrero  27 ',
                                    'Febrero  26','Febrero  26 ','Febrero  25','Febrero  25 ',
                                    'Febrero  24','Febrero  24 ','Febrero  23','Febrero  23 ',
                                    'Febrero  22','Febrero  22 ','Febrero  21','Febrero  21 ',
                                    'Febrero  20','Febrero  20 ','Febrero  19','Febrero  19 ',
                                    'Febrero  18','Febrero  18 ','Febrero  17','Febrero  17 ',
                                    'Febrero  16','Febrero  16 ','Febrero  15','Febrero  15 ',
                                    'Febrero  14','Febrero  14 ','Febrero  13','Febrero  13 ',
                                    'Febrero  12','Febrero  12 ','Febrero  11','Febrero  11 ',
                                    'Febrero  10','Febrero  10 ','Febrero  9 ','Febrero  9 ',
                                    'Febrero  8 ','Febrero  8  ','Febrero  7 ','Febrero  7 ',
                                    'Febrero  6 ','Febrero  6  ','Febrero  5 ','Febrero  5 ',
                                    'Febrero  4 ','Febrero  4  ','Febrero  3 ','Febrero  3 ',
                                    'Febrero  2 ','Febrero  2  ','Febrero  1 ','Febrero  1 ',
                                   
                                   'Marzo  31','Marzo  31 ', 'Marzo  30',
                                    'Marzo  30 ','Marzo  29','Marzo  29 ',
                                    'Marzo  28','Marzo  28 ','Marzo  27','Marzo  27 ',
                                    'Marzo  26','Marzo  26 ','Marzo  25','Marzo  25 ',
                                    'Marzo  24','Marzo  24 ','Marzo  23','Marzo  23 ',
                                    'Marzo  22','Marzo  22 ','Marzo  21','Marzo  21 ',
                                    'Marzo  20','Marzo  20 ','Marzo  19','Marzo  19 ',
                                    'Marzo  18','Marzo  18 ','Marzo  17','Marzo  17 ',
                                    'Marzo  16','Marzo  16 ','Marzo  15','Marzo  15 ',
                                    'Marzo  14','Marzo  14 ','Marzo  13','Marzo  13 ',
                                    'Marzo  12','Marzo  12 ','Marzo  11','Marzo  11 ',
                                    'Marzo  10','Marzo  10 ','Marzo  9 ','Marzo  9 ',
                                    'Marzo  8 ','Marzo  8  ','Marzo  7 ','Marzo  7 ',
                                    'Marzo  6 ','Marzo  6  ','Marzo  5 ','Marzo  5 ',
                                    'Marzo  4 ','Marzo  4  ','Marzo  3 ','Marzo  3 ',
                                    'Marzo  2 ','Marzo  2  ','Marzo  1 ','Marzo  1 ',
                                   
                                   'Abril  30','Abril  30 ','Abril  29','Abril  29 ',
                                    'Abril  28','Abril  28 ','Abril  27','Abril  27 ',
                                    'Abril  26','Abril  26 ','Abril  25','Abril  25 ',
                                    'Abril  24','Abril  24 ','Abril  23','Abril  23 ',
                                    'Abril  22','Abril  22 ','Abril  21','Abril  21 ',
                                    'Abril  20','Abril  20 ','Abril  19','Abril  19 ',
                                    'Abril  18','Abril  18 ','Abril  17','Abril  17 ',
                                    'Abril  16','Abril  16 ','Abril  15','Abril  15 ',
                                    'Abril  14','Abril  14 ','Abril  13','Abril  13 ',
                                    'Abril  12','Abril  12 ','Abril  11','Abril  11 ',
                                    'Abril  10','Abril  10 ','Abril  9 ','Abril  9 ',
                                    'Abril  8 ','Abril  8  ','Abril  7 ','Abril  7 ',
                                    'Abril  6 ','Abril  6  ','Abril  5 ','Abril  5 ',
                                    'Abril  4 ','Abril  4  ','Abril  3 ','Abril  3 ',
                                    'Abril  2 ','Abril  2  ','Abril  1 ','Abril  1 ',
                                   
                                   'Mayo  31','Mayo  31 ',
                                    'Mayo  30','Mayo  30 ','Mayo  29','Mayo  29 ',
                                    'Mayo  28','Mayo  28 ','Mayo  27','Mayo  27 ',
                                    'Mayo  26','Mayo  26 ','Mayo  25','Mayo  25 ',
                                    'Mayo  24','Mayo  24 ','Mayo  23','Mayo  23 ',
                                    'Mayo  22','Mayo  22 ','Mayo  21','Mayo  21 ',
                                    'Mayo  20','Mayo  20 ','Mayo  19','Mayo  19 ',
                                    'Mayo  18','Mayo  18 ','Mayo  17','Mayo  17 ',
                                    'Mayo  16','Mayo  16 ','Mayo  15','Mayo  15 ',
                                    'Mayo  14','Mayo  14 ','Mayo  13','Mayo  13 ',
                                    'Mayo  12','Mayo  12 ','Mayo  11','Mayo  11 ',
                                    'Mayo  10','Mayo  10 ','Mayo  9 ','Mayo  9 ',
                                    'Mayo  8 ','Mayo  8  ','Mayo  7 ','Mayo  7 ',
                                    'Mayo  6 ','Mayo  6  ','Mayo  5 ','Mayo  5 ',
                                    'Mayo  4 ','Mayo  4  ','Mayo  3 ','Mayo  3 ',
                                    'Mayo  2 ','Mayo  2  ','Mayo  1 ','Mayo  1 ',
                                   
                                   'Junio 30','Junio 30 ','Junio 29','Junio 29 ',
                                    'Junio 28','Junio 28 ','Junio 27','Junio 27 ',
                                    'Junio 26','Junio 26 ','Junio 25','Junio 25 ',
                                    'Junio 24','Junio 24 ','Junio 23','Junio 23 ',
                                    'Junio 22','Junio 22 ','Junio 21','Junio 21 ',
                                    'Junio 20','Junio 20 ','Junio 19','Junio 19 ',
                                    'Junio 18','Junio 18 ','Junio 17','Junio 17 ',
                                    'Junio 16','Junio 16 ','Junio 15','Junio 15 ',
                                    'Junio 14','Junio 14 ','Junio 13','Junio 13 ',
                                    'Junio 12','Junio 12 ','Junio 11','Junio 11 ',
                                    'Junio 10','Junio 10 ','Junio 9 ','Junio 9 ',
                                    'Junio 8 ','Junio 8  ','Junio 7 ','Junio 7 ',
                                    'Junio 6 ','Junio 6  ','Junio 5 ','Junio 5 ',
                                    'Junio 4 ','Junio 4  ','Junio 3 ','Junio 3 ',
                                    'Junio 2 ','Junio 2  ','Junio 1 ','Junio 1 ',
                                   
                                   'Julio 31','Julio 31','Julio 30','Julio 30 ','Julio 29','Julio 29 ',
                                    'Julio 28','Julio 28 ','Julio 27','Julio 27 ',
                                    'Julio 26','Julio 26 ','Julio 25','Julio 25 ',
                                    'Julio 24','Julio 24 ','Julio 23','Julio 23 ',
                                    'Julio 22','Julio 22 ','Julio 21','Julio 21 ',
                                    'Julio 20','Julio 20 ','Julio 19','Julio 19 ',
                                    'Julio 18','Julio 18 ','Julio 17','Julio 17 ',
                                    'Julio 16','Julio 16 ','Julio 15','Julio 15 ',
                                    'Julio 14','Julio 14 ','Julio 13','Julio 13 ',
                                    'Julio 12','Julio 12 ','Julio 11','Julio 11 ',
                                    'Julio 10','Julio 10 ','Julio 9 ','Julio 9 ',
                                    'Julio 8 ','Julio 8  ','Julio 7 ','Julio 7 ',
                                    'Julio 6 ','Julio 6  ','Julio 5 ','Julio 5 ',
                                    'Julio 4 ','Julio 4  ','Julio 3 ','Julio 3 ',
                                    'Julio 2 ','Julio 2  ','Julio 1 ','Julio 1 ',
                                   
                                   'Agosto 31','Agosto 31','Agosto 30','Agosto 30 ','Agosto 29','Agosto 29 ',
                                    'Agosto 28','Agosto 28 ','Agosto 27','Agosto 27 ',
                                    'Agosto 26','Agosto 26 ','Agosto 25','Agosto 25 ',
                                    'Agosto 24','Agosto 24 ','Agosto 23','Agosto 23 ',
                                    'Agosto 22','Agosto 22 ','Agosto 21','Agosto 21 ',
                                    'Agosto 20','Agosto 20 ','Agosto 19','Agosto 19 ',
                                    'Agosto 18','Agosto 18 ','Agosto 17','Agosto 17 ',
                                    'Agosto 16','Agosto 16 ','Agosto 15','Agosto 15 ',
                                    'Agosto 14','Agosto 14 ','Agosto 13','Agosto 13 ',
                                    'Agosto 12','Agosto 12 ','Agosto 11','Agosto 11 ',
                                    'Agosto 10','Agosto 10 ','Agosto 9 ','Agosto 9 ',
                                    'Agosto 8 ','Agosto 8  ','Agosto 7 ','Agosto 7 ',
                                    'Agosto 6 ','Agosto 6  ','Agosto 5 ','Agosto 5 ',
                                    'Agosto 4 ','Agosto 4  ','Agosto 3 ','Agosto 3 ',
                                    'Agosto 2 ','Agosto 2  ','Agosto 1 ','Agosto 1 ',
                                   
                                   'Septiembre 30','Septiembre 30 ','Septiembre 29','Septiembre 29 ',
                                    'Septiembre 28','Septiembre 28 ','Septiembre 27','Septiembre 27 ',
                                    'Septiembre 26','Septiembre 26 ','Septiembre 25','Septiembre 25 ',
                                    'Septiembre 24','Septiembre 24 ','Septiembre 23','Septiembre 23 ',
                                    'Septiembre 22','Septiembre 22 ','Septiembre 21','Septiembre 21 ',
                                    'Septiembre 20','Septiembre 20 ','Septiembre 19','Septiembre 19 ',
                                    'Septiembre 18','Septiembre 18 ','Septiembre 17','Septiembre 17 ',
                                    'Septiembre 16','Septiembre 16 ','Septiembre 15','Septiembre 15 ',
                                    'Septiembre 14','Septiembre 14 ','Septiembre 13','Septiembre 13 ',
                                    'Septiembre 12','Septiembre 12 ','Septiembre 11','Septiembre 11 ',
                                    'Septiembre 10','Septiembre 10 ','Septiembre 9 ','Septiembre 9 ',
                                    'Septiembre 8 ','Septiembre 8  ','Septiembre 7 ','Septiembre 7 ',
                                    'Septiembre 6 ','Septiembre 6  ','Septiembre 5 ','Septiembre 5 ',
                                    'Septiembre 4 ','Septiembre 4  ','Septiembre 3 ','Septiembre 3 ',
                                    'Septiembre 2 ','Septiembre 2  ','Septiembre 1 ','Septiembre 1 ',
                                   
                                   'Octubre 31','Octubre 31','Octubre 30','Octubre 30 ','Octubre 29','Octubre 29 ',
                                    'Octubre 28','Octubre 28 ','Octubre 27','Octubre 27 ',
                                    'Octubre 26','Octubre 26 ','Octubre 25','Octubre 25 ',
                                    'Octubre 24','Octubre 24 ','Octubre 23','Octubre 23 ',
                                    'Octubre 22','Octubre 22 ','Octubre 21','Octubre 21 ',
                                    'Octubre 20','Octubre 20 ','Octubre 19','Octubre 19 ',
                                    'Octubre 18','Octubre 18 ','Octubre 17','Octubre 17 ',
                                    'Octubre 16','Octubre 16 ','Octubre 15','Octubre 15 ',
                                    'Octubre 14','Octubre 14 ','Octubre 13','Octubre 13 ',
                                    'Octubre 12','Octubre 12 ','Octubre 11','Octubre 11 ',
                                    'Octubre 10','Octubre 10 ','Octubre 9 ','Octubre 9 ',
                                    'Octubre 8 ','Octubre 8  ','Octubre 7 ','Octubre 7 ',
                                    'Octubre 6 ','Octubre 6  ','Octubre 5 ','Octubre 5 ',
                                    'Octubre 4 ','Octubre 4  ','Octubre 3 ','Octubre 3 ',
                                    'Octubre 2 ','Octubre 2  ','Octubre 1 ','Octubre 1 ',
                                   
                                   'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                   
                                   'Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                   
                                   'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 '
                                   
                                   
                                  ]),
                                   
        'Valor E' :pd.Series ([24,28,26,27,28,27,28,27,28,22,28,22,22,25,24,25,21,27,28,24,28,23,25,27,
                            25,28,23,27,26,25,25,22,16,25,17,25,26,26,27,27,13,24,21,24,18,24,26,
                            27,28,26,20,27,28,13,16,25,22,22,23,27,28,24,#Diciembre 2020 
                           
                            28,25,18,28,24,8,19,26,28,25,19,16,23,27,26,25,22,23,24,26,22,26,
                            26,28,24,26,19,21,22,23,28,26,27,27,21,16,25,24,17,28,28,20,25,
                            27,27,18,21,27,23,16,28,12,24,26,27,17,10,24,26,26,#Noviembre 2020
                               
                            22,25,27,18,22,28,27,20,28,25,26,21,25,26,15,25,13,18,24,21,27,
                            25,24,22,27,21,17,24,24,25,27,23,16,22,27,28,28,25,25,27,22,28,24,21,26,21,
                            27,23,26,14,19,27,24,20,20,25,24,22,24,26,22,23,#Enero 2021
                            
                            27,25,27,19,24,25,23,21,24,24,24,27,23,25,26,23,23,19,27,25,28,28,15,27,28,24,25,
                            27,20,23,26,20,23,22,26,25,15,27,27,27,22,21,27,26,25,27,28,17,14,26,24,26,27,18,28,27, #Febrero 2021
                               
                            21,28,25,27,25,23,26,18,24,24,19,21,26,21,25,27,27,26,27,26,28,25,28,27,14,24,28,27,
                            23,28,28,21,28,23,26,28,23,22,28,26,19,20,23,27,21,16,24,26,17,21,27,26,25,27,28,20,
                            21,21,27,23,27,25, #Marzo 2021
                               
                            23,28,19,26,14,21,28,23,15,23,28,24,24,27,28,21,19,27,28,28,
                            19,24,28,27,28,26,26,18,26,15,27,21,28,22,21,24,27,20,22,22,23,28,25,24,13,28,27,
                            22,25,14,28,28,27,26,23,21,28,23,28,27, #Abril 2021
                               
                            20,25,26,27,28,19,24,23,23,27,27,19,27,27,24,27,26,24,26,26,18,16,15,
                            25,24,25,26,24,22,28,27,17,28,24,18,25,26,28,26,26,21,26,24,27,27,28,28,23,27,26,
                            25,23,17,22,21,28,23,22,24,28,28,21, #MAYO 2021
                               
                            25,27,18,27,24,23,14,24,20,21,22,27,23,28,25,27,27,27,28,27,28,26,
                            24,25,21,24,28,21,23,27,27,26,26,25,15,16,27,26,24,20,28,24,17,23,21,28,28,
                            13,25,17,26,22,25,27,24,25,26,25,20,26, #Junio 2021
                               
                            28,24,25,26,28,22,20,28,23,27,24,19,28,21,21,26,24,24,27,28,25,23,24,28,19,28,
                            26,22,26,26,20,24,25,25,26,22,18,20,27,27,11,27,24,25,25,28,26,18,27,27,25,13,
                               27,15,23,11,24,20,25,28,21,22, #JULIO E
                            
                            26,27,28,24,21,25,28,24,21,24,27,18,27,26,28,28,20,28,28,20,25,27,26,22,25,20,
                               26,28,28,25,26,26,28,25,21,25,27,28,25,27,20,19,27,23,19,27,21,24,25,26,28,23,
                               27,25,28,19,24,22,21,27,26,22, #AGOSTO E 
                               
                            25,27,22,21,29,26,22,18,27,20,28,24,26,22,23,28,26,27,23,26,26,
                               27,25,26,28,28,27,25,25,27,26,25,22,22,22,21,28,23,28,28,20,28,
                               17,17,21,27,25,23,24,25,27,25,28,26,28,23,23,27,23,22, #Septiembre E
                             
                            27,16,16,16,28,19,22,26,27,24,27,28,28,28,27,22,28,14,26,20,28,28,25,20,
                               28,25,22,28,28,24,19,26,26,27,26,12,28,28,28,26,23,16,26,20,22,21,28,
                               27,15,25,24,19,27,26,24,23,26,27,27,21,25,23,#october
                               
                            28,27,23,27,27,27,13,28,26,27,26,24,22,27,19,12,24,28,24,27,21,24,26,26,27,24,
                               26,22,22,26,28,24,27,24,15,28,23,21,26,27,26,25,27,19,22,21,27,22,27,26,23,23,23,
                               28,19,28,26,23,24,27, #November e
                               
                            26,24,27,23,28,28,26,21,18,25,21,28,24,24,20,25,24,25,26,25,
                               25,26,27,27,25,26,28,23,22,18,27,20,26,27,27,23,27,21,20,28,20,20,23,15,23,22,
                               288,27,23,17,13,27,18,21,25,24,26,26,23,26,12,8, #Diciembre E
                               
                               #ENERO E 2022
                               20,22,26,25,24,22,26,25,27,27,25,19,26,28,26,23,19,28,20,23
                             
                               
                            
                        
                            
                
                               
                               
                          ],
                          
                            index=['Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                   
                                    'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                    
                                    'Enero 31','Enero 31 ','Enero 30','Enero 30 ','Enero 29','Enero 29 ',
                                    'Enero 28','Enero 28 ','Enero 27','Enero 27 ',
                                    'Enero 26','Enero 26 ','Enero 25','Enero 25 ',
                                    'Enero 24','Enero 24 ','Enero 23','Enero 23 ',
                                    'Enero 22','Enero 22 ','Enero 21','Enero 21 ',
                                    'Enero 20','Enero 20 ','Enero 19','Enero 19 ',
                                    'Enero 18','Enero 18 ','Enero 17','Enero 17 ',
                                    'Enero 16','Enero 16 ','Enero 15','Enero 15 ',
                                    'Enero 14','Enero 14 ','Enero 13','Enero 13 ',
                                    'Enero 12','Enero 12 ','Enero 11','Enero 11 ',
                                    'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 ',
                                  
                                    'Febrero  28','Febrero  28 ','Febrero  27','Febrero  27 ',
                                    'Febrero  26','Febrero  26 ','Febrero  25','Febrero  25 ',
                                    'Febrero  24','Febrero  24 ','Febrero  23','Febrero  23 ',
                                    'Febrero  22','Febrero  22 ','Febrero  21','Febrero  21 ',
                                    'Febrero  20','Febrero  20 ','Febrero  19','Febrero  19 ',
                                    'Febrero  18','Febrero  18 ','Febrero  17','Febrero  17 ',
                                    'Febrero  16','Febrero  16 ','Febrero  15','Febrero  15 ',
                                    'Febrero  14','Febrero  14 ','Febrero  13','Febrero  13 ',
                                    'Febrero  12','Febrero  12 ','Febrero  11','Febrero  11 ',
                                    'Febrero  10','Febrero  10 ','Febrero  9 ','Febrero  9 ',
                                    'Febrero  8 ','Febrero  8  ','Febrero  7 ','Febrero  7 ',
                                    'Febrero  6 ','Febrero  6  ','Febrero  5 ','Febrero  5 ',
                                    'Febrero  4 ','Febrero  4  ','Febrero  3 ','Febrero  3 ',
                                    'Febrero  2 ','Febrero  2  ','Febrero  1 ','Febrero  1 ',
                                   
                                   'Marzo  31','Marzo  31 ', 'Marzo  30',
                                    'Marzo  30 ','Marzo  29','Marzo  29 ',
                                    'Marzo  28','Marzo  28 ','Marzo  27','Marzo  27 ',
                                    'Marzo  26','Marzo  26 ','Marzo  25','Marzo  25 ',
                                    'Marzo  24','Marzo  24 ','Marzo  23','Marzo  23 ',
                                    'Marzo  22','Marzo  22 ','Marzo  21','Marzo  21 ',
                                    'Marzo  20','Marzo  20 ','Marzo  19','Marzo  19 ',
                                    'Marzo  18','Marzo  18 ','Marzo  17','Marzo  17 ',
                                    'Marzo  16','Marzo  16 ','Marzo  15','Marzo  15 ',
                                    'Marzo  14','Marzo  14 ','Marzo  13','Marzo  13 ',
                                    'Marzo  12','Marzo  12 ','Marzo  11','Marzo  11 ',
                                    'Marzo  10','Marzo  10 ','Marzo  9 ','Marzo  9 ',
                                    'Marzo  8 ','Marzo  8  ','Marzo  7 ','Marzo  7 ',
                                    'Marzo  6 ','Marzo  6  ','Marzo  5 ','Marzo  5 ',
                                    'Marzo  4 ','Marzo  4  ','Marzo  3 ','Marzo  3 ',
                                    'Marzo  2 ','Marzo  2  ','Marzo  1 ','Marzo  1 ',
                                   
                                   'Abril  30','Abril  30 ','Abril  29','Abril  29 ',
                                    'Abril  28','Abril  28 ','Abril  27','Abril  27 ',
                                    'Abril  26','Abril  26 ','Abril  25','Abril  25 ',
                                    'Abril  24','Abril  24 ','Abril  23','Abril  23 ',
                                    'Abril  22','Abril  22 ','Abril  21','Abril  21 ',
                                    'Abril  20','Abril  20 ','Abril  19','Abril  19 ',
                                    'Abril  18','Abril  18 ','Abril  17','Abril  17 ',
                                    'Abril  16','Abril  16 ','Abril  15','Abril  15 ',
                                    'Abril  14','Abril  14 ','Abril  13','Abril  13 ',
                                    'Abril  12','Abril  12 ','Abril  11','Abril  11 ',
                                    'Abril  10','Abril  10 ','Abril  9 ','Abril  9 ',
                                    'Abril  8 ','Abril  8  ','Abril  7 ','Abril  7 ',
                                    'Abril  6 ','Abril  6  ','Abril  5 ','Abril  5 ',
                                    'Abril  4 ','Abril  4  ','Abril  3 ','Abril  3 ',
                                    'Abril  2 ','Abril  2  ','Abril  1 ','Abril  1 ',
                                   
                                   'Mayo  31','Mayo  31 ',
                                    'Mayo  30','Mayo  30 ','Mayo  29','Mayo  29 ',
                                    'Mayo  28','Mayo  28 ','Mayo  27','Mayo  27 ',
                                    'Mayo  26','Mayo  26 ','Mayo  25','Mayo  25 ',
                                    'Mayo  24','Mayo  24 ','Mayo  23','Mayo  23 ',
                                    'Mayo  22','Mayo  22 ','Mayo  21','Mayo  21 ',
                                    'Mayo  20','Mayo  20 ','Mayo  19','Mayo  19 ',
                                    'Mayo  18','Mayo  18 ','Mayo  17','Mayo  17 ',
                                    'Mayo  16','Mayo  16 ','Mayo  15','Mayo  15 ',
                                    'Mayo  14','Mayo  14 ','Mayo  13','Mayo  13 ',
                                    'Mayo  12','Mayo  12 ','Mayo  11','Mayo  11 ',
                                    'Mayo  10','Mayo  10 ','Mayo  9 ','Mayo  9 ',
                                    'Mayo  8 ','Mayo  8  ','Mayo  7 ','Mayo  7 ',
                                    'Mayo  6 ','Mayo  6  ','Mayo  5 ','Mayo  5 ',
                                    'Mayo  4 ','Mayo  4  ','Mayo  3 ','Mayo  3 ',
                                    'Mayo  2 ','Mayo  2  ','Mayo  1 ','Mayo  1 ',
                                   
                                   'Junio 30','Junio 30 ','Junio 29','Junio 29 ',
                                    'Junio 28','Junio 28 ','Junio 27','Junio 27 ',
                                    'Junio 26','Junio 26 ','Junio 25','Junio 25 ',
                                    'Junio 24','Junio 24 ','Junio 23','Junio 23 ',
                                    'Junio 22','Junio 22 ','Junio 21','Junio 21 ',
                                    'Junio 20','Junio 20 ','Junio 19','Junio 19 ',
                                    'Junio 18','Junio 18 ','Junio 17','Junio 17 ',
                                    'Junio 16','Junio 16 ','Junio 15','Junio 15 ',
                                    'Junio 14','Junio 14 ','Junio 13','Junio 13 ',
                                    'Junio 12','Junio 12 ','Junio 11','Junio 11 ',
                                    'Junio 10','Junio 10 ','Junio 9 ','Junio 9 ',
                                    'Junio 8 ','Junio 8  ','Junio 7 ','Junio 7 ',
                                    'Junio 6 ','Junio 6  ','Junio 5 ','Junio 5 ',
                                    'Junio 4 ','Junio 4  ','Junio 3 ','Junio 3 ',
                                    'Junio 2 ','Junio 2  ','Junio 1 ','Junio 1 ',
                                   
                                   'Julio 31','Julio 31','Julio 30','Julio 30 ','Julio 29','Julio 29 ',
                                    'Julio 28','Julio 28 ','Julio 27','Julio 27 ',
                                    'Julio 26','Julio 26 ','Julio 25','Julio 25 ',
                                    'Julio 24','Julio 24 ','Julio 23','Julio 23 ',
                                    'Julio 22','Julio 22 ','Julio 21','Julio 21 ',
                                    'Julio 20','Julio 20 ','Julio 19','Julio 19 ',
                                    'Julio 18','Julio 18 ','Julio 17','Julio 17 ',
                                    'Julio 16','Julio 16 ','Julio 15','Julio 15 ',
                                    'Julio 14','Julio 14 ','Julio 13','Julio 13 ',
                                    'Julio 12','Julio 12 ','Julio 11','Julio 11 ',
                                    'Julio 10','Julio 10 ','Julio 9 ','Julio 9 ',
                                    'Julio 8 ','Julio 8  ','Julio 7 ','Julio 7 ',
                                    'Julio 6 ','Julio 6  ','Julio 5 ','Julio 5 ',
                                    'Julio 4 ','Julio 4  ','Julio 3 ','Julio 3 ',
                                    'Julio 2 ','Julio 2  ','Julio 1 ','Julio 1 ',
                                   
                                   'Agosto 31','Agosto 31','Agosto 30','Agosto 30 ','Agosto 29','Agosto 29 ',
                                    'Agosto 28','Agosto 28 ','Agosto 27','Agosto 27 ',
                                    'Agosto 26','Agosto 26 ','Agosto 25','Agosto 25 ',
                                    'Agosto 24','Agosto 24 ','Agosto 23','Agosto 23 ',
                                    'Agosto 22','Agosto 22 ','Agosto 21','Agosto 21 ',
                                    'Agosto 20','Agosto 20 ','Agosto 19','Agosto 19 ',
                                    'Agosto 18','Agosto 18 ','Agosto 17','Agosto 17 ',
                                    'Agosto 16','Agosto 16 ','Agosto 15','Agosto 15 ',
                                    'Agosto 14','Agosto 14 ','Agosto 13','Agosto 13 ',
                                    'Agosto 12','Agosto 12 ','Agosto 11','Agosto 11 ',
                                    'Agosto 10','Agosto 10 ','Agosto 9 ','Agosto 9 ',
                                    'Agosto 8 ','Agosto 8  ','Agosto 7 ','Agosto 7 ',
                                    'Agosto 6 ','Agosto 6  ','Agosto 5 ','Agosto 5 ',
                                    'Agosto 4 ','Agosto 4  ','Agosto 3 ','Agosto 3 ',
                                    'Agosto 2 ','Agosto 2  ','Agosto 1 ','Agosto 1 ',
                                   
                                   'Septiembre 30','Septiembre 30 ','Septiembre 29','Septiembre 29 ',
                                    'Septiembre 28','Septiembre 28 ','Septiembre 27','Septiembre 27 ',
                                    'Septiembre 26','Septiembre 26 ','Septiembre 25','Septiembre 25 ',
                                    'Septiembre 24','Septiembre 24 ','Septiembre 23','Septiembre 23 ',
                                    'Septiembre 22','Septiembre 22 ','Septiembre 21','Septiembre 21 ',
                                    'Septiembre 20','Septiembre 20 ','Septiembre 19','Septiembre 19 ',
                                    'Septiembre 18','Septiembre 18 ','Septiembre 17','Septiembre 17 ',
                                    'Septiembre 16','Septiembre 16 ','Septiembre 15','Septiembre 15 ',
                                    'Septiembre 14','Septiembre 14 ','Septiembre 13','Septiembre 13 ',
                                    'Septiembre 12','Septiembre 12 ','Septiembre 11','Septiembre 11 ',
                                    'Septiembre 10','Septiembre 10 ','Septiembre 9 ','Septiembre 9 ',
                                    'Septiembre 8 ','Septiembre 8  ','Septiembre 7 ','Septiembre 7 ',
                                    'Septiembre 6 ','Septiembre 6  ','Septiembre 5 ','Septiembre 5 ',
                                    'Septiembre 4 ','Septiembre 4  ','Septiembre 3 ','Septiembre 3 ',
                                    'Septiembre 2 ','Septiembre 2  ','Septiembre 1 ','Septiembre 1 ',
                                   
                                   'Octubre 31','Octubre 31','Octubre 30','Octubre 30 ','Octubre 29','Octubre 29 ',
                                    'Octubre 28','Octubre 28 ','Octubre 27','Octubre 27 ',
                                    'Octubre 26','Octubre 26 ','Octubre 25','Octubre 25 ',
                                    'Octubre 24','Octubre 24 ','Octubre 23','Octubre 23 ',
                                    'Octubre 22','Octubre 22 ','Octubre 21','Octubre 21 ',
                                    'Octubre 20','Octubre 20 ','Octubre 19','Octubre 19 ',
                                    'Octubre 18','Octubre 18 ','Octubre 17','Octubre 17 ',
                                    'Octubre 16','Octubre 16 ','Octubre 15','Octubre 15 ',
                                    'Octubre 14','Octubre 14 ','Octubre 13','Octubre 13 ',
                                    'Octubre 12','Octubre 12 ','Octubre 11','Octubre 11 ',
                                    'Octubre 10','Octubre 10 ','Octubre 9 ','Octubre 9 ',
                                    'Octubre 8 ','Octubre 8  ','Octubre 7 ','Octubre 7 ',
                                    'Octubre 6 ','Octubre 6  ','Octubre 5 ','Octubre 5 ',
                                    'Octubre 4 ','Octubre 4  ','Octubre 3 ','Octubre 3 ',
                                    'Octubre 2 ','Octubre 2  ','Octubre 1 ','Octubre 1 ',
                                   
                                   'Noviembre 30','Noviembre 30 ','Noviembre 29','Noviembre 29 ',
                                    'Noviembre 28','Noviembre 28 ','Noviembre 27','Noviembre 27 ',
                                    'Noviembre 26','Noviembre 26 ','Noviembre 25','Noviembre 25 ',
                                    'Noviembre 24','Noviembre 24 ','Noviembre 23','Noviembre 23 ',
                                    'Noviembre 22','Noviembre 22 ','Noviembre 21','Noviembre 21 ',
                                    'Noviembre 20','Noviembre 20 ','Noviembre 19','Noviembre 19 ',
                                    'Noviembre 18','Noviembre 18 ','Noviembre 17','Noviembre 17 ',
                                    'Noviembre 16','Noviembre 16 ','Noviembre 15','Noviembre 15 ',
                                    'Noviembre 14','Noviembre 14 ','Noviembre 13','Noviembre 13 ',
                                    'Noviembre 12','Noviembre 12 ','Noviembre 11','Noviembre 11 ',
                                    'Noviembre 10','Noviembre 10 ','Noviembre 9 ','Noviembre 9 ',
                                    'Noviembre 8 ','Noviembre 8  ','Noviembre 7 ','Noviembre 7 ',
                                    'Noviembre 6 ','Noviembre 6  ','Noviembre 5 ','Noviembre 5 ',
                                    'Noviembre 4 ','Noviembre 4  ','Noviembre 3 ','Noviembre 3 ',
                                    'Noviembre 2 ','Noviembre 2  ','Noviembre 1 ','Noviembre 1 ',
                                   
                                   'Diciembre 31','Diciembre 31','Diciembre 30','Diciembre 30 ','Diciembre 29','Diciembre 29 ',
                                    'Diciembre 28','Diciembre 28 ','Diciembre 27','Diciembre 27 ',
                                    'Diciembre 26','Diciembre 26 ','Diciembre 25','Diciembre 25 ',
                                    'Diciembre 24','Diciembre 24 ','Diciembre 23','Diciembre 23 ',
                                    'Diciembre 22','Diciembre 22 ','Diciembre 21','Diciembre 21 ',
                                    'Diciembre 20','Diciembre 20 ','Diciembre 19','Diciembre 19 ',
                                    'Diciembre 18','Diciembre 18 ','Diciembre 17','Diciembre 17 ',
                                    'Diciembre 16','Diciembre 16 ','Diciembre 15','Diciembre 15 ',
                                    'Diciembre 14','Diciembre 14 ','Diciembre 13','Diciembre 13 ',
                                    'Diciembre 12','Diciembre 12 ','Diciembre 11','Diciembre 11 ',
                                    'Diciembre 10','Diciembre 10 ','Diciembre 9 ','Diciembre 9 ',
                                    'Diciembre 8 ','Diciembre 8  ','Diciembre 7 ','Diciembre 7 ',
                                    'Diciembre 6 ','Diciembre 6  ','Diciembre 5 ','Diciembre 5 ',
                                    'Diciembre 4 ','Diciembre 4  ','Diciembre 3 ','Diciembre 3 ',
                                    'Diciembre 2 ','Diciembre 2  ','Diciembre 1 ','Diciembre 1 ',
                                   
                                   'Enero 10','Enero 10 ','Enero 9 ','Enero 9 ',
                                    'Enero 8 ','Enero 8  ','Enero 7 ','Enero 7 ',
                                    'Enero 6 ','Enero 6  ','Enero 5 ','Enero 5 ',
                                    'Enero 4 ','Enero 4  ','Enero 3 ','Enero 3 ',
                                    'Enero 2 ','Enero 2  ','Enero 1 ','Enero 1 '
                                  
                                  ]),}
    
    df=pd.DataFrame(lote)
    print (df)
   
    
    salida=int(input("Pulse 10 para salir"))
    if salida ==10:
        exit()
    else:
        print("La opción NO está dentro de los parametros establecidos")
        

if opcion ==2:
    print("Esta opcion aún no esta lista")
    salida=int(input("Pulse 10 para salir"))
    if salida ==10:
        exit()
    else:
        print("La opción NO está dentro de los parametros establecidos")
    

    
    

if opcion ==3:
    print ("Para generar numeros elegidos por nuestro Algoritmo")
    f1=random.choice (["1","2","3","4","5"])
    f2=random.choice (["9","8","7","5","4"])
    f3=random.choice (["12","10","13","14","16"])
    f4=random.choice (["22","21","17","24","20"])
    f5=random.choice (["27","28","26","25","24"])

    f6=random.choice (["1","2","3","4","5"])
    f7=random.choice (["9","8","7","5","4"])
    f8=random.choice (["12","10","13","14","16"])
    f9=random.choice (["22","21","17","24","20"])
    f10=random.choice (["27","28","26","25","24"])

    f11=random.choice (["1","2","3","4","5"])
    f12=random.choice (["9","8","7","5","4"])
    f13=random.choice (["12","10","13","14","16"])
    f14=random.choice (["22","21","17","24","20"])
    f15=random.choice (["27","28","26","25","24"])

    f16=random.choice (["1","2","3","4","5"])
    f17=random.choice (["9","8","7","5","4"])
    f18=random.choice (["12","10","13","14","16"])
    f19=random.choice (["22","21","17","24","20"])
    f20=random.choice (["27","28","26","25","24"])

    f21=random.choice (["1","2","3","4","5"])
    f22=random.choice (["9","8","7","5","4"])
    f23=random.choice (["12","10","13","14","16"])
    f24=random.choice (["22","21","17","24","20"])
    f25=random.choice (["27","28","26","25","24"])

    f26=random.choice (["1","2","3","4","5"])
    f27=random.choice (["9","8","7","5","4"])
    f28=random.choice (["12","10","13","14","16"])
    f29=random.choice (["22","21","17","24","20"])
    f30=random.choice (["27","28","26","25","24"])

    f31=random.choice (["1","2","3","4","5"])
    f32=random.choice (["9","8","7","5","4"])
    f33=random.choice (["12","10","13","14","16"])
    f34=random.choice (["22","21","17","24","20"])
    f35=random.choice (["27","28","26","25","24"])

    f36=random.choice (["1","2","3","4","5"])
    f37=random.choice (["9","8","7","5","4"])
    f38=random.choice (["12","10","13","14","16"])
    f39=random.choice (["22","21","17","24","20"])
    f40=random.choice (["27","28","26","25","24"])

    f41=random.choice (["1","2","3","4","5"])
    f42=random.choice (["9","8","7","5","4"])
    f43=random.choice (["12","10","13","14","16"])
    f44=random.choice (["22","21","17","24","20"])
    f45=random.choice (["27","28","26","25","24"])

    f46=random.choice (["1","2","3","4","5"])
    f47=random.choice (["9","8","7","5","4"])
    f48=random.choice (["12","10","13","14","16"])
    f49=random.choice (["22","21","17","24","20"])
    f50=random.choice (["27","28","26","25","24"])
    aleatorio_array=np.array([f1,f2,f3,f4,f5])
    aleatorio_array2=np.array([f6,f7,f8,f9,f10])
    aleatorio_array3=np.array([f11,f12,f13,f14,f15])
    aleatorio_array4=np.array([f16,f17,f18,f19,f20])
    aleatorio_array5=np.array([f21,f22,f23,f24,f25])
    aleatorio_array6=np.array([f26,f27,f28,f29,f30])
    aleatorio_array7=np.array([f31,f32,f33,f34,f35])
    aleatorio_array8=np.array([f36,f37,f38,f39,f40])
    aleatorio_array9=np.array([f41,f42,f43,f44,f45])
    aleatorio_array10=np.array([f46,f47,f48,f49,f50])
    print (aleatorio_array)
    print (aleatorio_array2)
    print (aleatorio_array3)
    print (aleatorio_array4)
    print (aleatorio_array5)
    print (aleatorio_array6)
    print (aleatorio_array7)
    print (aleatorio_array8)
    print (aleatorio_array9)
    print (aleatorio_array10)
    
if opcion ==4:
    print ("Para generar Numeros sin nuestro algoritmo")

    G1=random.randrange(1,28)
    G2=random.randrange(1,28)
    G3=random.randrange(1,28)
    G4=random.randrange(1,28)
    G5=random.randrange(1,28)
    aleatorio_arrayAL=np.array([G1,G2,G3,G4,G5])
    print (aleatorio_arrayAL)

    G6=random.randrange(1,28)
    G7=random.randrange(1,28)
    G8=random.randrange(1,28)
    G9=random.randrange(1,28)
    G10=random.randrange(1,28)
    aleatorio_arrayAL2=np.array([G6,G7,G8,G9,G10])
    print (aleatorio_arrayAL2)

    G11=random.randrange(1,28)
    G12=random.randrange(1,28)
    G13=random.randrange(1,28)
    G14=random.randrange(1,28)
    G15=random.randrange(1,28)
    aleatorio_arrayAL3=np.array([G11,G12,G13,G14,G15])
    print (aleatorio_arrayAL3)

    G16=random.randrange(1,28)
    G17=random.randrange(1,28)
    G18=random.randrange(1,28)
    G19=random.randrange(1,28)
    G20=random.randrange(1,28)
    aleatorio_arrayAL4=np.array([G16,G17,G18,G19,G20])
    print (aleatorio_arrayAL4)

    G21=random.randrange(1,28)
    G22=random.randrange(1,28)
    G23=random.randrange(1,28)
    G24=random.randrange(1,28)
    G25=random.randrange(1,28)
    aleatorio_arrayAL5=np.array([G21,G22,G23,G24,G25])
    print (aleatorio_arrayAL5)

    G26=random.randrange(1,28)
    G27=random.randrange(1,28)
    G28=random.randrange(1,28)
    G29=random.randrange(1,28)
    G30=random.randrange(1,28)
    aleatorio_arrayAL6=np.array([G26,G27,G28,G29,G30])
    print (aleatorio_arrayAL6)

    G31=random.randrange(1,28)
    G32=random.randrange(1,28)
    G33=random.randrange(1,28)
    G34=random.randrange(1,28)
    G35=random.randrange(1,28)
    aleatorio_arrayAL7=np.array([G31,G32,G33,G34,G35])
    print (aleatorio_arrayAL7)

    G36=random.randrange(1,28)
    G37=random.randrange(1,28)
    G38=random.randrange(1,28)
    G39=random.randrange(1,28)
    G40=random.randrange(1,28)
    aleatorio_arrayAL8=np.array([G36,G37,G38,G39,G40])
    print (aleatorio_arrayAL8)

    G41=random.randrange(1,28)
    G42=random.randrange(1,28)
    G43=random.randrange(1,28)
    G44=random.randrange(1,28)
    G45=random.randrange(1,28)
    aleatorio_arrayAL9=np.array([G41,G42,G43,G44,G45])
    print (aleatorio_arrayAL9)

    G46=random.randrange(1,28)
    G47=random.randrange(1,28)
    G48=random.randrange(1,28)
    G49=random.randrange(1,28)
    G50=random.randrange(1,28)
    aleatorio_arrayAL10=np.array([G46,G47,G48,G49,G50])
    print (aleatorio_arrayAL10)

if opcion ==5:
    print ("***** Impresion de ambas listas *****")
    print ("***** Listas con nuestro algoritmo *****")
    print (aleatorio_array)
    print (aleatorio_array2)
    print (aleatorio_array3)
    print (aleatorio_array4)
    print (aleatorio_array5)
    print (aleatorio_array6)
    print (aleatorio_array7)
    print (aleatorio_array8)
    print (aleatorio_array9)
    print (aleatorio_array10)
    print (" ")
    print ("***** Listas SIN nuestro Algoritmo *****")
    print (aleatorio_arrayAL)
    print (aleatorio_arrayAL2)
    print (aleatorio_arrayAL3)
    print (aleatorio_arrayAL4)
    print (aleatorio_arrayAL5)
    print (aleatorio_arrayAL6)
    print (aleatorio_arrayAL7)
    print (aleatorio_arrayAL8)
    print (aleatorio_arrayAL9)
    print (aleatorio_arrayAL10)
    
    
if opcion ==6:
    print("Esta opcion aún no esta lista")
    salida=int(input("Pulse 10 para salir"))
    if salida ==10:
        exit()
    else:
        print("La opción NO está dentro de los parametros establecidos")

if opcion ==7:
    print ("***** Informacion General del arreglo *****")
    print ("A continuación se muestran algunos aspectos generales que nos llevó a tomar las decisiones en el algoritmo")
    print ("Informacion general de la Columna A")    
    df['Valor A'].describe() #Descripcion de la columna A
    
    
            


# In[ ]:





# In[ ]:




