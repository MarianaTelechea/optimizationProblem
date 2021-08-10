# optimizationProblem
Proyecto final para la materia de Investigación Operativa | Universidad de San Andrés. Problema de optimización.

El objetivo de este proyecto es analizar las entregas de 5 camionetas de una distribuidora, determinando la forma más óptima de entregar los pedidos (recorrido y el orden de entrega). Utilización de las coordenadas con Google Maps de las direcciones donde se hicieron las diferentes entregas al igual que el punto desde donde salen las camionetas. Con la librería de python de “Latitude longitude to distance” se transforman esos puntos a coordenadas en un gráfico de eje xy o a un valor de una distancia determinada

● Metodología de trabajo: Travelling Salesman Problem. Un repartidor que debe cumplir con una cierta cantidad de pedidos en un día y tiempo determinado, por lo que debemos encontrar la ruta más óptima.
● Variables de decisión: xij{ 1 Si la camioneta hace el recorrido de la dirección i a la dirección j, 0 en caso contrario
● Función objetivo → Distancias entre el punto de salida y cada uno de los diferentes puntos de entrega, al igual que las distancias entre cada punto de entrega.
  * Min Z = ∑i=0 ∑j=0 Dij.Xij 
● Restricciones → Tiempo máximo que puede trabajar un empleado, capacidad máxima de cada camioneta.
  * ∑i=0 Xij = 1
  * ∑j=0 Xij = 1
  * ui-uj+N.Xij ≤ N-1
