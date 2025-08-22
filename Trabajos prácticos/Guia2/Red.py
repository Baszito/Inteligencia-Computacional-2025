import numpy as np
import pandas as pandas

class Capa():
    def __init__(self, neuronas: int, neuronas_ant: int) -> None:
        self.W = np.zeros(neuronas, neuronas_ant)

        self.GradienteError = None
        #self.Y = y
        self.Deltas = None
        pass
    def GuardarY(self, y):
        self.y = y
    def f_ObtenerSalidas(self, y) -> list:
        pass
    def ObtenerGradiente(self, capa_siguiente: "Capa"):
        pass
    def ActualizarPesos(self, y) -> None:
        pass
    def DOT_W_DELTA(self) -> list:
        pass

class Red():
    # Procesa los datos crudos leídos por pandas y devuelve una tupla con x = [-1, x1, x2, ..., xn] e y=[y1, y2, ..., yn]
    def process_raw_data_(self, raw_data: pandas.DataFrame) -> tuple[np.array, np.array]:
        data = raw_data.to_numpy()
        y = data[:, -1]                            # Extrae la última columna de la matriz, que representa la salida.
        x = data[:, :-1]
        ones = -1*np.ones((data.shape[0], 1))      # Crea una columna de -1's
        x = np.concatenate([ones, x], axis=1)      # Concatena la columna de -1's
                                                   # cheat sheet rápido:
                                                   #   A[:, :-1] → todas las columnas menos la última
                                           
                                                   #   A[:, :-2] → todas las columnas menos las dos últimas
                                           
                                                   #   A[:, 1:] → todas las columnas menos la primera
        return (x, y)
    # Setea los datos de entrenamiento
    def set_training_path(self, path: str) -> None:
        y_t = self.process_raw_data_(pandas.read_csv(path, header=None))
        self.x_trn: np.ndarray = y_t[0]
        self.y_trn: np.ndarray = y_t[1]
    # Setea los datos de validación 
    def set_validation_path(self, path: str) -> None:
        y_t = self.process_raw_data_(pandas.read_csv(path, header=None))
        self.x_val: np.ndarray = y_t[0]
        self.y_val: np.ndarray = y_t[1]
    # Setea los datos de testing
    def set_testing_path(self, path: str) -> None:
        y_t = self.process_raw_data_(pandas.read_csv(path, header=None))
        self.x_tst: np.ndarray = y_t[0]
        self.y_tst: np.ndarray = y_t[1]
    def trn_rows(self):
        return self.y_trn.shape[0]
    def trn_cols(self):
        return self.x_trn.shape[1]
    def tes_rows(self):
        return self.y_tst.shape[0]
    def __init__(self, training_path: str, validation_path: str, testing_path: str, vector_capas: list, gamma: float = 0.01, epocas: int = 100, tasa_exito: float = 0.05):
        # Setea los datos.
        self.set_training_path(training_path)
        self.set_validation_path(validation_path)
        self.set_testing_path(testing_path)

        self.gamma = gamma
        self.epocas = epocas
        self.tasa_exito = tasa_exito

        self.numero_capas = len(vector_capas)
        self.capas = []
        print("Carga de datos completados")
        print(self.y_trn.shape[0])

        # Para la capa de entrada, asignar la cantidad de entradas como neuronas_ant.
        for i in range(1, self.numero_capas):

            capa = Capa(vector_capas[i], vector_capas[i-1])
            self.capas.append(capa)
        
        self.capas.append(Capa(vector_capas[self.numero_capas], 0))
    def Entrenar():
        # 1) Para cada epoca
        #   2) para cada patron
        #       3) Forward
        #       4) Backward
        #       5) Actualizacion
        #   3) Para cada patron
        #       4) Forward
        #       5) Calcular desempeño
        print("Entranamiento completado")
        pass
    



Red('Trabajos prácticos/Guia2/concent_trn.csv', 'Trabajos prácticos/Guia2/concent_trn.csv', 'Trabajos prácticos/Guia2/concent_tst.csv', [3, 3, 1], 0.01, 100, 0.95)
