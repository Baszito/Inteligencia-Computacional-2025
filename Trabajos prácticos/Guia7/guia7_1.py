import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

#------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------Algoritmo evolutivo, para comparacion----------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------------#
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
        self.posibilidad_mutacion=0.2
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
        return -f(individuo.bits_a_numero() - 512)
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

        #y actualizar al cant_genotipos
        self.cant_genotipos += 2
    
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
        if(it==self.max_iteraciones):
            print("No se llego a la aptitud requerida")
        else:
            print("Se llego a la aptitud requerida en : ")
            print(str(it))
        print("Solucion encontrada : ")
        #X
        print(self.mejor_individuo)
        #f(x)
        print("f(x)")
        print(f(self.mejor_individuo.bits_a_numero() - 512))
        print("x")
        print(self.mejor_individuo.bits_a_numero() - 512)
        print("Aptitud encontrada : ")
        print(self.mejor_aptitud)

#------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------Enjambre de particulas-------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------------#

def f(x):
    return -x[0]*math.sin(math.sqrt(abs(x[0])))        

class Particula:
    def __init__(self, dimensiones, min, max):
        #1 - Inicializacion
        self.x = [] #Posicion actual de la particula
        self.v=[]   #Velocidad actual de la particula
        self.y = [] #Mejor posicion de dicha particula
        for i in range(0, dimensiones):
            self.x.append(np.random.uniform(min, max))
            self.v.append(0)
        #print(self.x)
        self.y = self.x
    
    def actualizar_y(self):
        if (f(self.x) < f(self.y)):
            self.y= self.x
        
    def actualizar_pos(self, y_global, const1, const2,min,max):
        #Actualizar velocidad
        self.r = np.random.rand(2, len(self.x))
        self.v = np.array(self.v) + const1*self.r[0, :]*(np.array(self.y) - np.array(self.x)) + const2*self.r[1, :]*(np.array(y_global) - np.array(self.x))
        # Actualizar posicion
        #print(self.v)
        self.x= np.array(self.x) + np.array(self.v)
        self.x=np.clip(self.x,min,max)
                

class Enjambre:
    def __init__(self, cant_particulas, dimensiones, min, max, const1, const2):
        self.particulas = []
        self.y_global = []
        self.dimensiones = dimensiones
        self.cant_particulas = cant_particulas
        self.const1 = const1
        self.const2 = const2
        self.min = min
        self.max = max
        for i in range(0, cant_particulas):
            self.particulas.append(Particula(dimensiones, min, max))
            for j in range(0, dimensiones):
                self.y_global.append(0)
    def evaluar_particula(self, k):
        #print(self.particulas[k].x)
        #if f( self.particulas[k].x ) < f( self.particulas[k].y ):
        #    self.particulas[k].y = self.particulas[k].x
        
        if f( self.particulas[k].y ) < f( self.y_global ):
            print(self.particulas[k].y)
            self.y_global = self.particulas[k].y
    def loop_principal(self, max_it=1000, cuando_cortar = 0,crit_parada=10):
        it = 0
        se_llego = False
        it_global = 0
        while(it < max_it):
            y_gl_anterior = copy.deepcopy(self.y_global)
            for k in range(0, self.cant_particulas):
                #print(k)
                self.particulas[k].actualizar_y()
                self.evaluar_particula(k)
            for k in range(0, self.cant_particulas):
                self.particulas[k].actualizar_pos(self.y_global, self.const1, self.const2,self.min,self.max)
                print()
            if (y_gl_anterior == self.y_global):
                it_global += 1 
            else:
                it_global = 0
            
            if (it_global == crit_parada):
                se_llego = True
                break

            it += 1
        if (se_llego):
            print("Se llegó al valor deseado: ")
            print("Y Global (Mejor X)")
            print(str(self.y_global[0]))
            print("F(Y Global)")
            print(str(f(self.y_global)))
            print("Tomo un total de " + str(it) + " iteraciones")
        else:
            print("No se llego :(")
            print("Y Global (Mejor X)")
            print(str(self.y_global[0]))
            print("F(Y Global)")
            print(str(f(self.y_global)))
            print("Tomo un total de " + str(it) + " iteraciones")
            
        

        
        
iteraciones=2
cant_particulas=20
criterio_parada=400
#poblacion=AlgEvolutivo(20,20,criterio_parada,iteraciones)
#poblacion.evolucion()

x_enjambre=Enjambre(cant_particulas,1,-512,512,2,1)
x_enjambre.loop_principal(1000, -400)