import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

#   1. Inicializar poblacion (decodificacion) (10 bits)
#   2. mejorAptitud = evaluar(poblacion) (Funcion de fitness : 1/f(x))
#   3. mientras MejorAptitud < AptitudRequerida
       #3.1. Operador Seleccion
       #3.2. Operador Cruza
       #3.3. Operador Mutacion
#   4. fin



def fg(x):
    return -x*math.sin(math.sqrt(abs(x)))       
class Genotipo:
    def __init__(self, cant_bits):
        # Se inicializan los bits de manera aleatoria.
        self.cant_bits = cant_bits
        self.bits = []
        for i in range(0, cant_bits):
            self.bits.append(random.randint(0, 1))
    def bits_a_numero(self):
        """Convierte la lista de bits en un número entero."""
        numero = 0
        for bit in self.bits:
            numero = (numero << 1) | bit  # desplazar y agregar bit
        return numero
    def mutacion(self):
        self.posibilidad_mutacion=0.01
        self.aux=random.random()
        if(self.aux<=self.posibilidad_mutacion):
            randi = random.randint(0, self.cant_bits-1)
            if (self.bits[randi] == 1):
                self.bits[randi] = 0
            else:
                self.bits[randi] = 1
    def __str__(self):
        return "Soy el mejor individuo"
        

class AlgEvolutivo:
    def funcion_fitness_1(self, individuo: Genotipo):
        return -fg(individuo.bits_a_numero() - 512)
    def __init__(self,cant_genotipos,cant_bits, aptitud_requerida,max_iteraciones):
        self.poblacion=[]
        self.cant_genotipos=cant_genotipos
        self.cant_bits=cant_bits
        self.mejor_aptitud=0
        self.mejor_individuo = None
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
                self.mejor_individuo = self.poblacion[i]
        return (self.mejor_aptitud, self.mejor_individuo)
    
    def cruzar(self, padre1: Genotipo, padre2: Genotipo):
        self.posibilidad_cruza=0.8
        self.aux=random.random()

        hijo1 = padre1
        hijo2 = padre2
        if(self.aux<self.posibilidad_cruza):
            self.punto_cruza=padre1.cant_bits - random.randint(1,padre1.cant_bits-1)
            sublista1 = padre1.bits[0:self.punto_cruza]
            sublista2 = padre2.bits[self.punto_cruza:padre2.cant_bits]
            hijo1 = Genotipo(self.cant_bits)
            hijo1.bits = sublista1 + sublista2

            self.punto_cruza=padre1.cant_bits - random.randint(1,padre1.cant_bits-1)
            sublista1 = padre1.bits[0:self.punto_cruza]
            sublista2 = padre2.bits[self.punto_cruza:padre2.cant_bits]
            hijo2 = Genotipo(self.cant_bits)
            hijo2.bits = sublista1 + sublista2
        return (hijo1, hijo2)
        
    def seleccion(self):
        print('seleccion')
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
        self.numerito = random.randint(0, len(self.ruleta))
        return self.genotipos_ruleta[self.numerito]
    
    def seleccion_por_torneo(self, k = 3):
        #A las piñas muchachos
        #Seleccionamos k individuos al azar de la poblacion y su APTITUD
        seleccion = random.sample(list(zip(self.poblacion, self.aptitudes)), k)

        #Ordenamos los seleccionados de mayor a menor
        #key = lambda x:x[1] -> Usa como parametro de comparacion el 2do elemento de seleccion (la aptitud del individuo)
        seleccion.sort(key=lambda x: x[1], reverse=True)

        #Retornamos al individuo de mayor aptitud (el que quedo 1ero en el ordenamiento pue)
        return seleccion[0][0]

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
        
        print("...cargando...")
        it = 0
        self.evaluar_poblacion()
        for i in range(0, self.max_iteraciones):
            
            #print(str(it))

            self.generacion()
            self.evaluar_poblacion()
            
            it+=1
            if (self.mejor_aptitud >= self.aptitud_requerida):
                break
        if(it==self.max_iteraciones):
            print("No se llego a la aptitud requerida")
        else:
            print("Se llego a la aptitud requerida en " + str(it) + " iteraciones")
        print("Solucion encontrada : ")
        #X
        print(self.mejor_individuo)
        #f(x)
        print("f(x)")
        print(fg(self.mejor_individuo.bits_a_numero() - 512))
        print("x(Algoritmo evolutivo)")
        print(self.mejor_individuo.bits_a_numero() - 512)
        print("Aptitud encontrada : ")
        print(self.mejor_aptitud)
iteraciones=1000
poblacion=AlgEvolutivo(40,10,415,iteraciones)
poblacion.evolucion()

