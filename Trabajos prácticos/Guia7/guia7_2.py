import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy
import time

#------------Carga de datos y Armado de la matriz D------------#

#------------Inicializarion del Sistema------------#
#Inicializar parametros, hormigas y feromonas

#Loop
    #Loop por hormigas
        #Loop de probabilidad
    #Loop por conexion

#Si convergen al mismo camino, finaliza (Podemos cortar cuando las longitudes sean iguales o muy similares)
class AS:
    def __init__(self, archivo,cant_hormigas, c_inicio,tasa_evaporacion,esquema,alpha,beta,Q):
        self.cant_hormigas=cant_hormigas
        self.c_inicio=c_inicio
        self.tasa_evaporacion=tasa_evaporacion
        self.esquema=esquema
        self.alpha=alpha
        self.beta=beta
        self.Q=Q
        self.D = pd.read_csv(archivo, header=None).values
        self.cant_ciudades=self.D.shape[1]
        self.sigma=np.zeros((self.cant_ciudades,self.cant_ciudades))#matriz de feromonas
        self.hormigas=[]
        for i in range(0, self.cant_ciudades):
            for j in range(i+1, self.cant_ciudades):
                self.sigma[j, i]=np.random.uniform()
                self.sigma[i, j]=self.sigma[j, i]

        for i in range(cant_hormigas):
            self.hormigas.append(c_inicio)
        
        self.caminos={}
        self.caminos_viejo = {}
        #self.caminos = []
        #for h in self.hormigas:
        #    self.caminos.append([])
        
    def distancia_total(self, camino):  
        largo = 0
        for i in range(0, len(camino)-1):  
            c1 = camino[i]
            c2 = camino[i+1]
            
            largo += self.D[c1, c2]
            
        return largo
            
    def elegir_siguiente(self, hormiga, c_visitadas):
        #print(hormiga)
        
        denom = 0
        for i in range (0,self.cant_ciudades):
            if(np.isin(i, c_visitadas) or i == hormiga):
                pass
            else:
                feromon = self.sigma[hormiga, i]
                deseo_moverse = 1/self.D[hormiga, i]
                denom += (feromon**self.alpha)*(deseo_moverse**self.beta)

        probabilidades = []
        
        for i in range(0, self.cant_ciudades):
            if(np.isin(i, c_visitadas) or i == hormiga):
                probabilidades.append(0)
            else:
                if (denom != 0):
                    feromon = self.sigma[hormiga, i]
                    deseo_moverse = 1/self.D[hormiga, i]
                    prob = (feromon**self.alpha)*(deseo_moverse**self.beta)
                    probabilidades.append(prob/denom)
                else:
                    probabilidades.append(0)
        # k es el nro de elementos
        #print(probabilidades)
        return random.choices(range(len(probabilidades)), weights=probabilidades, k=1)[0]
                        
        
    def actualizar_feromonas(self, esquema):
        #Evaporacion
        for i in range(self.cant_ciudades):
            for j in range(self.cant_ciudades):
                if(i!=j):
                    self.sigma[i,j]=(1-self.tasa_evaporacion)*self.sigma[i,j]
       
        #Depositar feromonas
        for k in range(self.cant_hormigas):
            delta=0
            camino=self.caminos[k]
            for indice in range(len(camino)-1):
                i=camino[indice]
                j=camino[indice+1]
                if(esquema==0):         #esquema Global
                    delta = self.Q/self.distancia_total(camino)
                if(esquema==1):         #esquema uniforme
                    delta = self.Q
                if(esquema==2):         #esquema local
                    delta = self.Q/self.D[i,j]
                self.sigma[i,j] += delta
                self.sigma[j,i] += delta #La matriz tiene que ser simetrica
            

            
    def construir_camino(self, hormiga):

        c_hormiga = self.hormigas[hormiga] 
        c_hormiga = int(c_hormiga)

        p_hormiga = []
        p_hormiga.append(c_hormiga)

        for i in range(0, self.cant_ciudades-1):
            n_ciudad = self.elegir_siguiente(c_hormiga, p_hormiga)
            c_hormiga = n_ciudad
            p_hormiga.append(n_ciudad)
        p_hormiga.append(self.c_inicio)
        return p_hormiga
        
             
    def bucle_principal(self):
        self.convergio_camino=False
        it = 0
        while (self.convergio_camino==False):
            for k in range(self.cant_hormigas):#bucle de hormigas
                #nuevo_camino=np.zeros(self.cant_ciudades)
                nuevo_camino=self.construir_camino(k)

                # Una vez encontrado el destino se ejecuta la siguiente línea
                self.caminos[k] = nuevo_camino
            self.actualizar_feromonas(self.esquema)    
            it += 1
            if (self.caminos == self.caminos_viejo):
                self.convergio_camino = True

            self.caminos_viejo = copy.deepcopy(self.caminos)
            #for i in range(self.cant_ciudades):#bucle de conexiones
            #    for j in range(self.cant_ciudades):
            #        if(i!=j): 
            #for k in range(self.cant_hormigas-1):
             #   if((self.caminos[k]==self.caminos[k+1]).all):
        self.buscar_camino_minimo()
        print("Cantidad de iteraciones :" + str(it))
    def buscar_camino_minimo(self):
        min = 689234768924
        i = 0
        inmin = -1
        for k in range(len(self.caminos)):
            if self.distancia_total(self.caminos[k]) < min:
                min = self.distancia_total(self.caminos[k])
                inmin = i
            i+=1
        print("La cantidad de kilometros recorrida fue de " + str(min) + "km")
        print("El camino tomado fue: ")
        print(self.caminos[inmin])



        # Esq: 0 -> Global
        #      1 -> Uniforme
        #      2 -> Local
hormiguero = AS("Trabajos prácticos\Guia7\gr17.csv", cant_hormigas=10, c_inicio=5, tasa_evaporacion=0.2, esquema=2, alpha=2, beta=2, Q=8)

hormiguero.bucle_principal()