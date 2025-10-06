import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

#1. Inicializar poblacion (decodificacion) (10 bits)
#2. mejorAptitud = evaluar(poblacion) (Funcion de fitness : 1/f(x))
#3. mientras MejorAptitud < AptitudRequerida
    #3.1. Operador Seleccion
    #3.2. Operador Cruza
    #3.3. Operador Mutacion
#4. fin

# 16 bits, 8 bits para x, 8 bits para y.

class Genotipo:
    def __init__(self, cant_bits):
        # Se inicializan los bits de manera aleatoria.
        self.cant_bits = cant_bits
        self.bits = []
        for i in range(0, cant_bits):
            self.bits.append(random.randint(0, 1))
        self.clamp_binario()
    def bits_a_numero(self):
        """Convierte la lista de bits en un número entero."""
        numero = 0
        for bit in self.bits:
            numero = (numero << 1) | bit  # desplazar y agregar bit
        return numero
    
    def get_x(self):
        return self.bits[0:8]
    #Mapeo lineal entre 0 y 256 de -100 a 100
    def get_x_numero(self):
        x = self.get_x()
        numero = 0
        for bit in x:
            numero = (numero << 1) | bit  # desplazar y agregar bit
        return numero
    def get_y_numero(self):
        y = self.get_y()
        numero = 0
        for bit in y:
            numero = (numero << 1) | bit  # desplazar y agregar bit
        return numero
    def get_y(self):
        return self.bits[8:16]
    def clamp_binario(self):
        if(self.get_y_numero()>200):
        #11001000
            self.bits[8:16]=[1, 1, 0, 0, 1, 0, 0, 0]
        if(self.get_x_numero()>200):
            self.bits[0:8]=[1, 1, 0, 0, 1, 0, 0, 0]
            
    def mutacion(self):
        self.posibilidad_mutacion_x=0.01
        self.posibilidad_mutacion_y=0.01
        self.aux_x=random.random()
        if(self.aux_x<=self.posibilidad_mutacion_x):
            randi = random.randint(0, 7)
            if (self.bits[randi] == 1):
                self.bits[randi] = 0
            else:
                self.bits[randi] = 1
        self.aux_y=random.random()
        if(self.aux_y<=self.posibilidad_mutacion_y):
            randi = random.randint(8, 15)
            if (self.bits[randi] == 1):
                self.bits[randi] = 0
            else:
                self.bits[randi] = 1


        #self.posibilidad_mutacion = 0.1
        #self.aux_x=random.random()
        #if(self.aux_x<=self.posibilidad_mutacion):
        #    randi = random.randint(0, 15)
        #    if (self.bits[randi] == 1):
        #        self.bits[randi] = 0
        #    else:
        #        self.bits[randi] = 1
        self.clamp_binario()
            
    def __str__(self):
        return "Soy el mejor individuo"
        
def f(x, y):
    return ((x**2 + y**2)**(0.25))*((math.sin(50*((((x**2) + (y**2))**0.1)))**2) + 1)
def d_f(x, y): #para utilizar el metodo del gradiente
    # #if x == 0:  # evitar división por cero
    # #    print("GRADIENTE INVALIDO")
    # #    return float('nan')
    # #return -math.sin(math.sqrt(abs(x))) - (x**2 * math.cos(math.sqrt(abs(x)))) / (2 * abs(x) * math.sqrt(abs(x)))
    # d = (((x**2 + y**2)**(0.25)) * (( math.sin( 50 * ( (x**2 + y**2) ** 0.1 ) )**2 ) + 1))
    s = x**2 + y**2
    if s == 0:
        return (0.0, 0.0)  # gradiente indefinido, devolvemos 0

    dfdx = (0.5 * x * (math.sin(50 * (s**0.1))**2 + 1) * (s**-0.75)) + (20 * x * math.cos( 50 * (( s )**0.1) ) * math.sin( 50 * ((s)**0.1)) * ((s)**-0.65))#10 * A * (s**-0.9) * math.sin(2 * u)
    dfdy = (0.5 * y * (math.sin(50 * (s**0.1))**2 + 1) * (s**-0.75)) + (20 * y * math.cos( 50 * (( s )**0.1) ) * math.sin( 50 * ((s)**0.1)) * ((s)**-0.65))#10 * A * (s**-0.9) * math.sin(2 * u)
    #dfdx = x * factor
    #dfdy = y * factor
    return (dfdx, dfdy)
    #return (d, d)

# Punto inicial debe ser de dos dimensiones
def metodo_gradiente_descendiente(punto_inicial, iteraciones, gamma):
    x, y = punto_inicial
    for i in range(0, iteraciones):
        dx, dy = d_f(x, y)
        x = x - gamma*dx
        y = y - gamma*dy
        #p = p - gamma*d_f(p)
        # forzar rango [-100, 100]
        if x > 100:
            x = 100
        if x < -100:
            x = -100

        if y > 100:
            y = 100
        if y < -100:
            y = -100
    return (x, y)

def gradiente_descendiente_global(punto_minimo,punto_maximo,cant_puntos,gamma,iteraciones):
    mejor_x=100
    mejor_y=100
    for i in range(0,cant_puntos):
        nuevo_x,nuevo_y=metodo_gradiente_descendiente((random.uniform(punto_minimo,punto_maximo), random.uniform(punto_minimo,punto_maximo)),iteraciones,gamma)
        if(f(mejor_x, mejor_y)>f(nuevo_x,nuevo_y)):
            mejor_x=nuevo_x
            mejor_y=nuevo_y
    return mejor_x, mejor_y

class AlgEvolutivo:
    def funcion_fitness_1(self, individuo: Genotipo):
        return -f(individuo.get_x_numero() - 100, individuo.get_y_numero() - 100)
    def __init__(self,cant_genotipos,cant_bits, aptitud_requerida,max_iteraciones):
        self.poblacion=[]
        self.cant_genotipos=cant_genotipos
        self.cant_bits=cant_bits
        self.mejor_aptitud=-99999999
        self.peor_aptitud1=99999999
        self.peor_aptitud2=99999999
        
        self.mejor_individuo = None
        self.peor_individuo1 = -1
        self.peor_individuo2 = -1
        self.aptitud_requerida = aptitud_requerida
        self.aptitudes = []
        self.max_iteraciones = max_iteraciones
        #Inicializo poblacion 
        for i in range(0,cant_genotipos):
            self.poblacion.append(Genotipo(self.cant_bits))

    def evaluar_poblacion(self):
        for i in range(0,self.cant_genotipos):
            aptitud_actual=self.funcion_fitness_1(self.poblacion[i])
            self.aptitudes.append(aptitud_actual)
            if(self.mejor_aptitud<aptitud_actual):
                self.mejor_aptitud=aptitud_actual
                self.mejor_individuo = copy.deepcopy(self.poblacion[i])

            if(self.peor_aptitud1>aptitud_actual):
                self.peor_aptitud2=self.peor_aptitud1
                self.peor_aptitud1=aptitud_actual
                self.peor_individuo2=self.peor_individuo1
                self.peor_individuo1=i
            elif(self.peor_aptitud2>aptitud_actual):
                self.peor_aptitud2=aptitud_actual
                self.peor_individuo2=i
        return (self.mejor_aptitud, self.mejor_individuo)
    
    def cruzar(self, padre1: Genotipo, padre2: Genotipo):
        self.posibilidad_cruza=0.8
        self.aux=random.random()

        hijo1 = padre1
        hijo2 = padre2
        if(self.aux<self.posibilidad_cruza):
            self.punto_cruza_x= len(padre1.get_x()) - random.randint(1,len(padre1.get_x())-1)
            self.punto_cruza_y= len(padre1.get_y()) - random.randint(1,len(padre1.get_y())-1)
            sublista1_x = padre1.get_x()[0:self.punto_cruza_x]
            sublista2_x = padre2.get_x()[self.punto_cruza_x:len(padre1.get_x())]

            sublista1_y = padre1.get_y()[0:self.punto_cruza_y]
            sublista2_y = padre2.get_y()[self.punto_cruza_y:len(padre1.get_y())]

            
            hijo1 = Genotipo(self.cant_bits)
            hijo1.bits = sublista1_x + sublista2_x + sublista1_y + sublista2_y

            self.punto_cruza_x= len(padre1.get_x()) - random.randint(1,len(padre1.get_x())-1)
            self.punto_cruza_y= len(padre1.get_y()) - random.randint(1,len(padre1.get_y())-1)
            sublista1_x = padre1.get_x()[0:self.punto_cruza_x]
            sublista2_x = padre2.get_x()[self.punto_cruza_x:len(padre1.get_x())]

            sublista1_y = padre1.get_y()[0:self.punto_cruza_y]
            sublista2_y = padre2.get_y()[self.punto_cruza_y:len(padre1.get_y())]

            
            hijo2 = Genotipo(self.cant_bits)
            hijo2.bits = sublista1_x + sublista2_x + sublista1_y + sublista2_y

            hijo1.clamp_binario()
            hijo2.clamp_binario()
            #self.punto_cruza=padre1.cant_bits - random.randint(1,padre1.cant_bits-1)
            #sublista1 = padre1.bits[0:self.punto_cruza]
            #sublista2 = padre2.bits[self.punto_cruza:padre2.cant_bits]
            #hijo2 = Genotipo(self.cant_bits)
            #hijo2.bits = sublista1 + sublista2
        return (hijo1, hijo2)
    
    def seleccion_por_torneo(self, k = 3):
        #A las piñas muchachos
        #Seleccionamos k individuos al azar de la poblacion y su APTITUD
        seleccion = random.sample(list(zip(self.poblacion, self.aptitudes)), k)

        #Ordenamos los seleccionados de mayor a menor
        #key = lambda x:x[1] -> Usa como parametro de comparacion el 2do elemento de seleccion (la aptitud del individuo)
        seleccion.sort(key=lambda x: x[1], reverse=True)

        #Retornamos al individuo de mayor aptitud (el que quedo 1ero en el ordenamiento pue)
        return seleccion[0][0]
        
    def seleccion(self):
        #Se implementa algoritmo de ruleta
        #Sumas todos
        #Sacas un numero entre 0 y la cantidad maxima
        #
        aptitud_max = 0
        aptitud_min = 9999999
        for i in range(0, self.cant_genotipos):
            if (self.aptitudes[i] < aptitud_min):
                aptitud_min = self.aptitudes[i]
            aptitud_max += self.aptitudes[i]
        #self.probabilidades = []
        #for i in range(0, self.cant_genotipos):
        #    self.probabilidades.append(self.aptitudes[i]/aptitud_total)
        self.genotipos_ruleta = []
        self.ruleta = []
        for i in range(0, self.cant_genotipos):
            apt = aptitud_max - aptitud_min
            tam = math.floor(self.aptitudes[i]*apt)
            for j in range(0, tam):
                self.ruleta.append(self.aptitudes[i])
                self.genotipos_ruleta.append(self.poblacion[i])
        self.numerito = random.randint(0, len(self.ruleta) - 1)
        return self.genotipos_ruleta[self.numerito]
    
    def generacion(self):
        poblacion_aux=[]
        for i in range(0, int(self.cant_genotipos/2)):
            #elegir 2 padres
            #self.progenitor1=self.seleccion()
            #self.progenitor2=self.seleccion()

            self.progenitor1=self.seleccion_por_torneo()
            self.progenitor2=self.seleccion_por_torneo()
            
            #cruzarlos
            hijo1, hijo2 = self.cruzar(self.progenitor1, self.progenitor2)
            
            #mutarlo
            hijo1.mutacion()
            hijo2.mutacion()

            #agregarlo al arreglo de la poblacion
            poblacion_aux.append(hijo1)
            poblacion_aux.append(hijo2)

        self.poblacion=poblacion_aux
        
    
    def evolucion(self):
        print("Algoritmo Evolutivo : ")
        #print("Iteraciones :")
        it = 0
        self.evaluar_poblacion()
        
        for i in range(0, self.max_iteraciones):
            
            #print(str(it))
            
            self.generacion()
            self.evaluar_poblacion()
            
            it+=1

            #print("x: ")
            #print(self.mejor_individuo.get_x_numero()-100)
            #print("y: ")
            #print(self.mejor_individuo.get_y_numero()-100)
            #print("f(x,y)")
            #print(f(self.mejor_individuo.get_x_numero()-100, self.mejor_individuo.get_y_numero()-100))
            if (self.mejor_aptitud >= self.aptitud_requerida):
                break
            #if (abs(self.mejor_aptitud) < abs(self.aptitud_requerida)):
            #    break
        if(it==self.max_iteraciones):
            print("No se llego a la aptitud requerida")
        else:
            print("Se llego a la aptitud requerida en : ")
            print(str(it))
        print("Solucion encontrada : ")
        print(self.mejor_individuo)
        print("x: ")
        print(self.mejor_individuo.get_x_numero()-100)
        print("y: ")
        print(self.mejor_individuo.get_y_numero()-100)
        print("f(x, y)")
        print(f(self.mejor_individuo.get_x_numero()-100, self.mejor_individuo.get_y_numero()-100))
        print("Aptitud encontrada : ")
        print(-self.mejor_aptitud)

iteraciones=1000
individuos_inicial=50
poblacion=AlgEvolutivo(individuos_inicial,16,-0.5,iteraciones)
poblacion.evolucion()

mejor_x, mejor_y=gradiente_descendiente_global(-100,100,individuos_inicial,0.5,iteraciones)
print("Metodo gradiente: ")
print("X")
print(mejor_x)
print("Y")
print(mejor_y)

print("F(X, Y)")
print(f(mejor_x, mejor_y))
