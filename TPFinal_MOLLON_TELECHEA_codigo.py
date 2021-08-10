#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 07:51:05 2020

@author: marianatelechea
"""

## TP FINAL

import numpy as np
from python_tsp.distances import great_circle_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.heuristics import solve_tsp_local_search
import picos


# =============================================================================
#                    PRUEBA DE CÓDIGO DE 5 PUNTOS
# =============================================================================

#EJEMPLO CON CINCO PUNTOS DENTRO DE UN CÍRCULO PARA PROBAR QUE FUNCIONE EL CÓDIGO
import picos
P = picos.Problem()
P.options.verbosity = 1
x = picos.BinaryVariable("x", (5, 5))
u = picos.RealVariable("u", 6, lower = 0)
M = 99999
c = [M, 1.1756, 1.9021, 1.9021, 1.1756,
     1.1756, M, 1.1756, 1.9021, 1.9021,
     1.9021, 1.1756, M, 1.1756, 1.9021,
     1.9021, 1.9021, 1.1756, M, 1.1756,
     1.1756, 1.9021, 1.9021, 1.1756, M]
N = 5
c = picos.Constant("c", c)
P.set_objective("min", c.T*x.reshaped((N**2, 1)))
for i in range(1, N):
    for j in range(1, N):
        if (i!=j):
             P.add_constraint(u[i+1] - u[j+1] + N*x[i, j] <= N-1)
for i in range(N):
    P.add_constraint(picos.sum(x[i,:])==1)
for j in range(N):
    P.add_constraint(picos.sum(x[:,j])==1)
print(P)
P.solve(solver="glpk")
print(np.array(x.value))

# Restricciones escritas a mano, antes de hacer un for loop.
# =============================================================================
# #Siendo ui - uj + NXij ≤ N - 1  i ≠ j; i = 1,2,3,...,N, j = 1,2,3,...,N
# #Flechas que salen del nodo 2
# P.add_constraint(u[2] - u[3] + 5*x[7] <= 4)
# P.add_constraint(u[2] - u[4] + 5*x[8] <= 4)
# P.add_constraint(u[2] - u[5] + 5*x[9] <= 4)
# #Flechas que salen del nodo 3
# P.add_constraint(u[3] - u[2] + 5*x[11] <= 4)
# P.add_constraint(u[3] - u[4] + 5*x[13] <= 4)
# P.add_constraint(u[3] - u[5] + 5*x[14] <= 4)
# #Flechas que salen del nodo 4
# P.add_constraint(u[4] - u[2] + 5*x[16] <= 4)
# P.add_constraint(u[4] - u[3] + 5*x[17] <= 4)
# P.add_constraint(u[4] - u[5] + 5*x[19] <= 4)
# #Flechas que salen del nodo 5
# P.add_constraint(u[5] - u[2] + 5*x[21] <= 4)
# P.add_constraint(u[5] - u[3] + 5*x[22] <= 4)
# P.add_constraint(u[5] - u[4] + 5*x[23] <= 4)
# #Siendo ∑(i=0) Xij = 1
# P.add_constraint(x[0]   + x[1]   + x[2]   + x[3]    + x[4]   == 1)
# P.add_constraint(x[5]   + x[6]   + x[7]   + x[8]    + x[9]   == 1)
# P.add_constraint(x[10]  + x[11]  + x[12]  + x[13]   + x[14]  == 1)
# P.add_constraint(x[15]  + x[16]  + x[17]  + x[18]   + x[19]  == 1)
# P.add_constraint(x[20]  + x[21]  + x[22]  + x[23]   + x[24]  == 1)
# #Siendo ∑(j=0) Xij = 1
# P.add_constraint(x[0]   + x[5]  + x[10]  + x[15]   + x[20]   == 1)
# P.add_constraint(x[1]   + x[6]  + x[11]  + x[16]   + x[21]   == 1)
# P.add_constraint(x[2]   + x[7]  + x[12]  + x[17]   + x[22]   == 1)
# P.add_constraint(x[3]   + x[8]  + x[13]  + x[18]   + x[23]   == 1)
# P.add_constraint(x[4]   + x[9]  + x[14]  + x[19]   + x[24]   == 1)
# =============================================================================

# Comienzo de problemas de camionetas

# =============================================================================
#                           DISTANCIAS EUCLIDEAN
# =============================================================================

# =============================================================================
#                              CAMIONETA - RENE
# =============================================================================


# A partir de las latitudes y longitudes, buscamos las distancias de cada camioneta

sources = np.array([
    [-34.5189821, -58.5244123], # Estado inicial
    [-34.5796668, -58.5034596], # Pedido 1
    [-34.5786717, -58.4914565], # Pedido 2
    [-34.5973516, -58.4549777], # Pedido 3
    [-34.5746675, -58.4883943], # Pedido 4
    [-34.57419,   -58.4888041], # Pedido 5
    [-34.5640789, -58.5061923], # Pedido 6
    [-34.5758684, -58.4927562], # Pedido 7
    [-34.5731341, -58.4862318]  # Pedido 8
])

distance_matrix = great_circle_distance_matrix(sources) # Calcula las distancias

print(np.round(distance_matrix)) # Nos devuelve una matrix 9x9

# Código para usamos para corroborar que nos esta dando la resolución correcta
# =============================================================================
# permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
# 
# print(permutation)
# print(distance)
# =============================================================================


P = picos.Problem()
P.options.verbosity = 1
x = picos.BinaryVariable("x", (9,9))
u = picos.RealVariable("u", 10, lower = 0)
M = 99999
c = [   M,  7015,  7291, 10787,  7016,  6951,  5285,  6958,  6963,
     7015,     M,  1104,  4854,  1487,  1474,  1751,  1067,  1737,
     7291,  1104,     M,  3933,   526,   554,  2110,   334,   780,
     10787, 4854,  3933,     M,  3965,  4028, 5973,   4203,  3929,
     7016,  1487,   526,  3965,     M,    65,  2010,   421,   261,
     6951,  1474,   554,  4028,    65,     M,  1949,   407,   263,
     5285,  1751,  2110,  5973,  2010,  1949,     M,  1798,  2087,
     6958,  1067,   334,  4203,   421,   407,  1798,     M,   670,
     6963,  1737,   780,  3929,   261,   263,  2087,   670,     M]
N = 9
c = picos.Constant("c", c)
P.set_objective("min", c.T*x.reshaped((N**2, 1)))

for i in range(1, N):
    for j in range(1, N):
        if (i!=j):
             P.add_constraint(u[i+1] - u[j+1] + N*x[i, j] <= N-1)
             
for i in range(N):
    P.add_constraint(picos.sum(x[i,:])==1)
for j in range(N):
    P.add_constraint(picos.sum(x[:,j])==1)
print(P)
P.solve(solver="glpk")
print(np.array(x.value))
print("La distancia en metros es " + str(P.value)) # Distancia óptima


# Óptimo que tenemos que llegar
# [0, 6, 1, 7, 2, 3, 8, 4, 5] // Recorrido
# 23576.284030312792 // Distancia

# Recorrido real = [0, 3, 1, 2, 7, 4, 5, 8, 6, 0]

def convierte_distancia(metros):
	mi = metros * 0.000621371192  
	return mi

distMetro = P.value

distMil = convierte_distancia(distMetro)
print("La distancia en millas es " + str(distMil))

# 14.649447222592 // Distancia óptima (Euclidean)
# 15.6 // Distancia real (Euclidean)




# =============================================================================
#                           CAMIONETA - EZEQUIEL
# =============================================================================

sources = np.array([
    [-34.5189821, -58.5244123], # Estado inicial
    [-34.679064,-58.5588573],
    [-34.7516205,-58.5934278],
    [-34.6852859,-58.5657153],
    [-34.6828322,-58.5786404],
    [-34.6794805,-58.5711405],
    [-34.7504571,-58.5938731],
    [-34.674607,-58.5713263],
    [-34.7321561,-58.5750604],
    [-34.7754201,-58.6512637],
    [-34.6766605,-58.5699102],
    [-34.6883602,-58.5655229] # Pedido 11
])
distance_matrix = great_circle_distance_matrix(sources)

print(np.round(distance_matrix))

# Código para usamos para corroborar que nos esta dando la resolución correcta
# =============================================================================
# permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
# 
# print(permutation)
# print(distance)
# =============================================================================

P = picos.Problem()
P.options.verbosity = 1
x = picos.BinaryVariable("x", (12,12))
u = picos.RealVariable("u", 13, lower = 0)
M = 99999
c = [    M, 18077, 26628, 18875, 18883, 18352, 26512, 17830, 24153, 30785, 18021, 19206,
     18077,     M,  8665,   934,  1857,  1124,  8559,  1243,  6087, 13642,  1045,  1200,
     26628,  8665,     M,  7799,  7767,  8276,   136,  8799,  2739,  5909,  8608,  7482,
     18875,   934,  7799,     M,  1213,   814,  7690,  1294,  5281, 12711,  1033,   342,
     18883,  1857,  7767,  1213,     M,   781,  7647,  1133,  5494, 12249,  1053,  1348,
     18352,  1124,  8276,   814,   781,     M,  8161,   542,  5868, 12939,   333,  1113,
     26512,  8559,   136,  7690,  7647,  8161,     M,  8682,  2664,  5932,  8493,  7375, 
     17830,  1243,  8799,  1294,  1133,   542,  8682,     M,  6408, 13380,   263,  1619,
     24153,  6087,  2739,  5281,  5494,  5868,  2664,  6408,     M,  8462,  6189,  4947,
     30785, 13642,  5909, 12711, 12249, 12939,  5932, 13380,  8462,     M, 13262, 12454,
     18021,  1045,  8608,  1033,  1053,   333,  8493,   263,  6189, 13262,     M,  1361,
     19206,  1200,  7482,   342,  1348,  1113,  7375,  1619,  4947, 12454,  1361,     M]
N = 12
c = picos.Constant("c", c)
P.set_objective("min", c.T*x.reshaped((N**2, 1)))

for i in range(1, N):
    for j in range(1, N):
        if (i!=j):
             P.add_constraint(u[i+1] - u[j+1] + N*x[i, j] <= N-1)
             
for i in range(N):
    P.add_constraint(picos.sum(x[i,:])==1)
for j in range(N):
    P.add_constraint(picos.sum(x[:,j])==1)
print(P)
P.solve(solver="glpk")
print(np.array(x.value))
print("La distancia en metros es " + str(P.value)) # Distancia óptima



# [0, 7, 10, 5, 4, 9, 2, 6, 8, 11, 3, 1, 0] // Recorrido óptimo (Euclidean)

# [0, 1, 7, 10, 5, 4, 3, 11, 9, 2, 6, 8, 0] // Recorrido real

def convierte_distancia(metros):
	mi = metros * 0.000621371192  
	return mi

distMetro = P.value

distMil = convierte_distancia(distMetro)
print("La distancia en millas es " + str(distMil))

# 40.05669389228 // Distancia óptima (Euclidean)
# 40 // Distancia real (Euclidean)


# =============================================================================
#                      OTRAS CAMIONETAS - EUCLIDEAN
# =============================================================================

# =============================================================================
#                             CAMIONETA AQUILES
# =============================================================================

sources = np.array([
    [-34.5189821, -58.5244123], # Estado inicial
    [-34.606495, -58.3853453],
    [-34.6026012, -58.3831427],
    [-34.605264,-58.3797583],
    [-34.6049241,-58.3885707],
    [-34.605335,-58.3875773],
    [-34.6046279,-58.3845841],
    [-34.6057538,-58.386872],
    [-34.6058119,-58.3775377],
    [-34.6036149,-58.3882073],
    [-34.6049679,-58.3841226],
    [-34.6046361,-58.3888728],
    [-34.6038088,-58.3883312],
    [-34.6033415,-58.3897447]
])
distance_matrix = great_circle_distance_matrix(sources)

print(np.round(distance_matrix))

permutation, distance = solve_tsp_dynamic_programming(distance_matrix) # Resultado de manera exacta 
permutation2, distance2 = solve_tsp_simulated_annealing(distance_matrix) # Método aproximado
permutation3, distance3 = solve_tsp_local_search(distance_matrix) # Método aproximado

print('permutation = ' + str(permutation))
print('distance = ' + str(distance))
print('permutation2 = ' + str(permutation2))
print('distance2 = ' + str(distance2))
print('permutation3 = ' + str(permutation3))
print('distance3 = ' + str(distance3)) 

def convierte_distancia(metros):
	mi = metros * 0.000621371192  
	return mi

distMetro = distance

distMil = convierte_distancia(distMetro)
print("La distancia en millas es " + str(distMil))


distMetro = distance2

distMil2 = convierte_distancia(distMetro)
print("La distancia en millas es " + str(distMil2))


distMetro = distance3

distMil3 = convierte_distancia(distMetro)
print("La distancia en millas es " + str(distMil3))


# [0, 2, 8, 3, 10, 6, 1, 7, 5, 4, 11, 12, 9, 13] // Recorrido exacto
# 20.87644122010789 // Distancia exacta

# [0, 2, 8, 3, 10, 6, 1, 7, 5, 4, 11, 12, 9, 13] // Recorrido aproximado 01
# 20.87644122010789 // Distancia aproximada 01

# [13, 12, 9, 6, 10, 2, 3, 8, 1, 7, 5, 4, 11, 0] // Recorrido aproximado 02
# 20.927111753467763 // Distancia aproximada 02


# Recorrido real = [0, 9, 12, 11, 4, 5, 7, 8, 1, 6, 10, 3, 13, 2]
# 21,6 // Distancia real


# =============================================================================
#                             CAMIONETA ADRIAN B.
# =============================================================================

sources = np.array([
    [-34.5189821, -58.5244123], # Estado inicial
    [-34.6491741, -58.6332878],
    [-34.7616631, -58.3960918],
    [-34.8280246, -58.3905235],
    [-34.8599442, -58.3889581],
    [-34.8321424,  -58.414613],
    [-34.8434995, -58.3758752],
    [-34.797167,  -58.4249471],
    [-34.8053288,   -58.44281],
    [-34.8189355, -58.3970388],
    [-34.8349562, -58.4034457],
    [-34.8618644,  -58.392049],
    [-34.8391173, -58.3838948],
    [-34.7785096, -58.4612266],
    [-34.817675,  -58.4326673],
    [-34.8125247, -58.4232752],
    [-34.9482607, -58.3595938],
    [-34.8124762, -58.3816328],
    [-34.8214795, -58.3936875],
    [-34.9170762, -58.3800598] # Pedido 19
])
distance_matrix = great_circle_distance_matrix(sources)

print(np.round(distance_matrix))

permutation, distance = solve_tsp_local_search(distance_matrix) # Método aproximado

print(permutation)
print(distance)

# Recorrido óptimo = [0, 2, 17, 9, 18, 3, 4, 11, 19, 16, 6, 12, 10, 5, 15, 14, 8, 7, 13, 1]
# Recorrido real =   [0, 1, 13, 8, 15, 14, 7, 2, 17, 9, 18, 3, 5, 10, 12, 6, 4, 11, 19, 16]

def convierte_distancia(metros):
	mi = metros * 0.000621371192  
	return mi

distMetro = distance

distMil = convierte_distancia(distMetro)

print("La distancia en millas es " + str(distMil))
print("Recorrido óptimo es " + str(permutation))

# Dsitancia óptima: 73.26737576016899
# Distancia real: 113



# =============================================================================
#                             CAMIONETA VICTOR
# =============================================================================

sources = np.array([
    [-34.5189821, -58.5244123], # Estado inicial
    [-34.5576896,-58.4494796],
    [-34.5503762,-58.4694796],
    [-34.5626816,-58.4524837],
    [-34.5620171,-58.4591584],
    [-34.5495813,-58.4701465],
    [-34.557824,-58.4628274],
    [-34.5402482,-58.4767271],
    [-34.5602107,-58.460862],
    [-34.5663926,-58.4546997],
    [-34.5523182,-58.5053508],
    [-34.5721363,-58.4450521],
    [-34.5479146,-58.4714137],
    [-34.5568173,-58.4637716],
    [-34.5624601,-58.4641877],
    [-34.5701956,-58.4458597],
    [-34.5660306,-58.4415207],
    [-34.5638296,-58.4587657],
    [-34.5670146,-58.4600907],
    [-34.5577023,-58.4693235],
    [-34.5714865,-58.4435239],
    [-34.5591199,-58.4637276],
    [-34.5550976,-58.4647737],
    [-34.5626806,-58.4599297],
    [-34.5659146,-58.4558607],
    [-34.5698204,-58.4439522]  # Pedido 25
])
distance_matrix = great_circle_distance_matrix(sources)

print(np.round(distance_matrix))


permutation, distance = solve_tsp_local_search(distance_matrix) # Método aproximado

print(permutation)
print(distance)

def convierte_distancia(metros):
	mi = metros * 0.000621371192  
	return mi

distMetro = distance

distMil = convierte_distancia(distMetro)

print("La distancia en millas es " + str(distMil))
print("Recorrido óptimo es " + str(permutation))

# Distancia óptima: 14.13547041325294
# Distancia real: 28.2

    
# Recorrido óptimo = [0, 10, 19, 14, 18, 9, 24, 17, 23, 4, 3, 15, 11, 20, 25, 16, 1, 8, 21, 6, 13, 22, 2, 5, 12, 7]
# Recorrido real   = [0, 16, 25, 20, 11, 15, 9, 24, 3, 18, 17, 23, 14, 1, 4, 8, 21, 6, 13, 19, 22, 2, 5, 12, 7, 10]


###############################################################################
###############################################################################
###############################################################################

# =============================================================================
#                           DISTANCIAS MANHATTAN
# =============================================================================


# =============================================================================
#                              CAMIONETA - RENE 
# =============================================================================

P = picos.Problem()
P.options.verbosity = 1
x = picos.BinaryVariable("x", (9,9))
u = picos.RealVariable("u", 10, lower = 0)
M = 99999
c = [ M,	5.7,	5.6,	8.0,	5.3,	5.2,	5.6,	5.8,	5.1,
     6.1,	 M,   	0.9,	3.7,	1.1,	1.1,	1.8,	1.2,	1.8,
     6.8,	1.3,     M,   	3.7,	0.9,	0.8,	2.5,	0.5,	1.1,     
     8.1,	4.1,	3.7,	 M,	    3.8,	3.7,	5.7,	3.2,	3.7,
     5.7,	1.2,	0.6,	3.3,	 M,	    0.6,	1.7,	0.6,	0.6,
     6.1,	1.1,	0.5,	3.4,	0.04,	 M,	    1.6,	0.6,	0.7,
     4.7,	1.4,	1.6,	4.9,	1.5,	1.5,	 M,	    1.9,	2.2,
     5.7,	1.0,	0.4,	3.7,	0.4,	0.4,	1.5,	 M,	    1.1,
     5.3,	1.2,	0.6,	3.9,	0.2,	0.4,	1.8,	0.7,	 M]
N = 9
c = picos.Constant("c", c)
P.set_objective("min", c.T*x.reshaped((N**2, 1)))

for i in range(1, N):
    for j in range(1, N):
        if (i!=j):
             P.add_constraint(u[i+1] - u[j+1] + N*x[i, j] <= N-1)
             
for i in range(N):
    P.add_constraint(picos.sum(x[i,:])==1)
for j in range(N):
    P.add_constraint(picos.sum(x[:,j])==1)
print(P)
P.solve(solver="glpk")
print(np.array(x.value))
print("La distancia en millas es " + str(P.value)) # Distancia óptima

# [0, 6, 1, 2, 8, 3, 4, 5, 9, 0] // Recorrido óptimo 

# [0, 3, 1, 2, 7, 4, 5, 8, 6, 0] // Recorrido real


# 20.240000000000002 // Distancia óptima
# 21.6 // Distancia real

# =============================================================================
#                         CAMIONETA - EZEQUIEL 
# =============================================================================

P = picos.Problem()
P.options.verbosity = 1
x = picos.BinaryVariable("x", (12,12))
u = picos.RealVariable("u", 13, lower = 0)
M = 99999
c = [   M, 13.7,   22.6,	14.8,  14.2,   13.7,	22.5,	13.4,	 21,  	27.4,  13.6,	 15,
     14.1,	  M,	 8, 	 1.1,	1.4,	  1, 	 8.3,	   1,	5.4,	  10,	  1,	1.9,
     22.6,    8,	 M,    	 7.5,	7.9,	  8,	 0.5,	 7.4,	2.3,	 5.1,	7.5,	7.1,
     14.6, 	0.5,	7.8,	   M,   1.2,	0.8,	 6.8,	 1.3,	  5,     9.9,	1.3,	1.8,
     14.5,	1.8,	8.4,	 1.3,	  M,      1,	 6.8,	 0.8,	4.9,	 9.8,	0.9,	1.3,
     14.2,	1.1,	8.5,	   1,	0.6,	  M,	 7.6,	 0.5,	5.8,	  10,	0.6,	1.4,
     22.7,	7.9,	0.4,	 6.9,	7.8,	6.4,	   M,	 8.2,	2.2,	 5.3,	8.3,	7.9,
     13.7,	 1,  	8.6,	 1.1,	0.8,	0.5,	 8.7,  	   M,	  6,	10.2,	0.2,	 15,
     21.5,	5.5,	2.6,	   5,	5.4,	5.5,	 2.1,	 6.2,	  M,	 7.4,	6.3,	  6,
     26.7,	9.6,	4.1,	   9,	9.5,	9.4,	   4,	 9.9,	6.2,	   M,	6.3,	  6,
     13.7,	0.9,	8.4,	 0.9,	0.9,	0.3,	 7.6,	 0.3,	5.8,	 5.8,	  M,    1.3,
     21.2,	1.7,	8,	     1.2,	1.7,	1.9,	 6.2,	 2.5,	4.4,	 4.4,	2.7,	  M]
N = 12
c = picos.Constant("c", c)
P.set_objective("min", c.T*x.reshaped((N**2, 1)))

for i in range(1, N):
    for j in range(1, N):
        if (i!=j):
             P.add_constraint(u[i+1] - u[j+1] + N*x[i, j] <= N-1)
             
for i in range(N):
    P.add_constraint(picos.sum(x[i,:])==1)
for j in range(N):
    P.add_constraint(picos.sum(x[:,j])==1)
print(P)
P.solve(solver="glpk")
print(np.array(x.value))
print("La distancia en millas es " + str(P.value)) # Distancia óptima

# [0, 1, 3, 8, 2, 6, 9, 11, 4, 5, 10, 7, 0] // Recorrido óptimo

# [0, 1, 7, 10, 5, 4, 3, 11, 9, 2, 6, 8, 0] // Recorrido real


# 46.5 // Distancia óptima
# 57.2 // Distancia real


# =============================================================================
#                MEJORAS A FUTURO - AUTOMZATIZACIÓN
# =============================================================================

#import picos
#from geopy.geocoders import Nominatim
#import numpy as np
#from python_tsp.distances import great_circle_distance_matrix
#from python_tsp.exact import solve_tsp_dynamic_programming
#geolocator = Nominatim(user_agent="Spyder")

#CANTIDAD DE DIRECCIONES (INCLUYENDO EL PUNTO DE SALIDA)
#N = 9
#PUNTO DE SALIDA
#ubicacion0 = geolocator.geocode("Panamá 3340, Munro, Buenos Aires")
#PUNTOS DE ENTREGAS

#ubicacion1 = geolocator.geocode("General Jose Artigas 5180, Villa Pueyrredon, Argentina")
#ubicacion2 = geolocator.geocode("ALTOLAGUIRRE 2199, VILLA URQUIZA, Buenos Aires")
#ubicacion3 = geolocator.geocode("Coronel Antonio Susini 2223, Buenos Aires")
#ubicacion4 = geolocator.geocode("AVENIDA TRIUNVIRATO 4664, VILLA URQUIZA, Argentina")
#ubicacion5 = geolocator.geocode("AV TRIUNVIRATO 4710, VILLA URQUIZA, Buenos Aires")
#ubicacion6 = geolocator.geocode("Rogelio Yrurtia 6030, Buenos Aires, Argentina")
#ubicacion7 = geolocator.geocode("BUCARELLI 2293, Buenos Aires, Argentina")
#ubicacion8 = geolocator.geocode("MONROE 4900, VILLA URQUIZA, Buenos Aires")
#lat_long0 = ubicacion0.latitude, ubicacion0.longitude
#lat_long1 = ubicacion1.latitude, ubicacion1.longitude
#lat_long2 = ubicacion2.latitude, ubicacion2.longitude
#lat_long3 = ubicacion3.latitude, ubicacion3.longitude
#lat_long4 = ubicacion4.latitude, ubicacion4.longitude
#lat_long5 = ubicacion5.latitude, ubicacion5.longitude
#lat_long6 = ubicacion6.latitude, ubicacion6.longitude
#lat_long7 = ubicacion7.latitude, ubicacion7.longitude
#lat_long8 = ubicacion8.latitude, ubicacion8.longitude
#sources = np.array([
#    lat_long0,
#    lat_long1,
#    lat_long2,
#    lat_long3,
#    lat_long4,
#    lat_long5,
#    lat_long6,
#    lat_long7,
#    lat_long8
#])
#distance_matrix = great_circle_distance_matrix(sources)
#print(np.round(distance_matrix))
#permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
#c = distance_matrix.reshape(N**2)
#P = picos.Problem()
#P.options.verbosity = 1
#x = picos.BinaryVariable("x", (9, 9))
#u = picos.RealVariable("u", 10, lower = 0)
#c = picos.Constant("c", c)
#P.set_objective("min", c.T*x.reshaped((N**2, 1)))
#for i in range(1, N):
#    for j in range(1, N):
#        if (i!=j):
#             P.add_constraint(u[i+1] - u[j+1] + N*x[i, j] <= N-1)
#for i in range(N):
#    P.add_constraint(picos.sum(x[i,:])==1)
#for j in range(N):
#    P.add_constraint(picos.sum(x[:,j])==1)
#for i in range (0, N**2, N+1):
#    P.add_constraint(x[i] == 0)
#print(P)
#P.solve(solver="glpk")
#print(np.array(x.value))
#print(" ")
#print ("El recorrido que se debe hacer es " + str(permutation))
#print("La distancia total recorrida es " + str(np.round(distance)))



