import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

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
        return self.bits[0:7]
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
        return self.bits[7:16]
    def clamp_binario(self):
        if(self.get_y_numero()>200):
        #11001000
            self.bits[8:15]=[1, 1, 0, 0, 1, 0, 0, 0]
        if(self.get_x_numero()>200):
            self.bits[0:7]=[1, 1, 0, 0, 1, 0, 0, 0]
            
    def mutacion(self):
        self.posibilidad_mutacion_x=0.2
        self.posibilidad_mutacion_y=0.2
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
        self.clamp_binario()
            
    def __str__(self):
        return "Soy el mejor individuo"
        
def f(x, y):
    return ((x**2 + y**2)**(0.25))*((math.sin(50*(((x**2) + (y**2)**0.1)))**2) + 1)
def d_f(x): #para utilizar el metodo del gradiente
    if x == 0:  # evitar división por cero
        print("GRADIENTE INVALIDO")
        return float('nan')
    return -math.sin(math.sqrt(abs(x))) - (x**2 * math.cos(math.sqrt(abs(x)))) / (2 * abs(x) * math.sqrt(abs(x)))
def metodo_gradiente_descendiente(punto_inicial, iteraciones, gamma):
    p = punto_inicial
    for i in range(0, iteraciones):
        p = p - gamma*d_f(p)
        # forzar rango [-512, 512]
        if p < -512:
            p = -512
        elif p > 512:
            p = 512
    return p
def gradiente_descendiente_global(punto_minimo,punto_maximo,cant_puntos,gamma,iteraciones):
    puntos=[]
    mejor_x=0
    for i in range(0,cant_puntos):
        puntos.append(metodo_gradiente_descendiente(random.uniform(punto_minimo,punto_maximo),iteraciones,gamma))
        if(f(mejor_x)>f(puntos[i])):
            mejor_x=puntos[i]
    return mejor_x

class AlgEvolutivo:
    def funcion_fitness_1(self, individuo: Genotipo):
        return 5/f(individuo.get_x_numero() - 100, individuo.get_y_numero() - 100)
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
                self.mejor_individuo = self.poblacion[i]
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
        self.numerito = random.randint(0, len(self.ruleta))
        return self.genotipos_ruleta[self.numerito]
    
    def generacion(self):
        #elegir 2 padres
        self.progenitor1=self.seleccion()
        self.progenitor2=self.seleccion()
        
        #cruzarlos
        hijo1, hijo2 = self.cruzar(self.progenitor1, self.progenitor2)
        
        #mutarlo
        hijo1.mutacion()
        hijo2.mutacion()

        #agregarlo al arreglo de la poblacion
        self.poblacion.append(hijo1)
        self.poblacion.append(hijo2)

        del self.poblacion[self.peor_individuo1]
        del self.poblacion[self.peor_individuo2]
        #y actualizar al cant_genotipos
        #self.cant_genotipos += 2
        
    
    def evolucion(self):
        print("Algoritmo Evolutivo : ")
        print("Iteraciones :")
        it = 0
        self.evaluar_poblacion()
        for i in range(0, self.max_iteraciones):
            
            print(str(it))

            self.generacion()
            self.evaluar_poblacion()
            
            it+=1
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
        #X
        print(self.mejor_individuo)
        #f(x)
        print("x: ")
        print(self.mejor_individuo.get_x_numero()-100)
        print("y: ")
        print(self.mejor_individuo.get_y_numero()-100)
        print("f(x, y)")
        print(f(self.mejor_individuo.get_x_numero()-100, self.mejor_individuo.get_y_numero()-100))
        print("Aptitud encontrada : ")
        print(self.mejor_aptitud)

iteraciones=2000
#La mejor aptitud que se puede lograr con numeros enteros es 4.677966171082638
poblacion=AlgEvolutivo(500,16,135871387,iteraciones)
poblacion.evolucion()

#gdgx=gradiente_descendiente_global(-512,512,20,0.5,iteraciones)
#print("Metodo gradiente: ")
#print("X")
#print(gdgx)
#print("f(x)")
#print(f(gdgx))