from abc import ABC, abstractmethod
import pandas as pandas
import numpy as np

# Clase abstracta que procesa los datos, cada perceptrón debería heredar de esta clase.
class Int(ABC):
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
    def cols(self):
        return self.x_trn.shape[1]
    def tes_rows(self):
        return self.y_tst.shape[0]
    # Constructor Wrapper de las dos anteriores, que aprovecha para inicializar los datos.
    def __init__(self, training_path: str, validation_path: str, testing_path: str, gamma: float = 0.01, max_iterations: int = 100, success_rate: float = 0.05) -> None:
        # Setea los datos.
        self.set_training_path(training_path)
        self.set_validation_path(validation_path)
        self.set_testing_path(testing_path)

        self.gamma = gamma
        self.max_iterations = max_iterations
        self.success_rate_criteria = success_rate
        self.w = np.random.rand(1, self.cols()) - 0.5*np.ones((1, self.cols())) # Genera el vector de w's
        self.w = self.w.ravel()

    # El siguiente método abstracto es para definir la función sigmoide.
    @abstractmethod
    def sigmoid(self, v):
        pass
        
    # El siguiente método abstracto define el w(n+1)
    @abstractmethod
    def update_w(self, x_datos, y_calculado, y_real):
        pass
    
    # El siguiente método define la salida de datos.
    def output(self, x) -> float:
        return self.sigmoid((np.dot(self.w, x)))

    # El siguiente método devuelve el ratio de éxito de nuestro predictor.
    def _success_rate_from_data(self, x_data, y_data) -> float:
        success_rate: float = 0
        rows = x_data.shape[0]
        for j in range(0, rows):
            y_real = y_data[j]
            y_calculado = self.output(x_data[j]) # Consigue el y(n)
            if int(y_calculado) == int(y_real):
                success_rate += 1.0
        return success_rate/rows
    def success_rate(self, what_data: str) -> float:
        x_data = None
        y_data = None
        if what_data.lower() == "trn":
            x_data = self.x_trn
            y_data = self.y_trn
        elif what_data.lower() == "val":
            x_data = self.x_val
            y_data = self.y_val
        elif what_data.lower() == "tst":
            x_data = self.x_tst
            y_data = self.y_tst
        else:
            raise ValueError("El argumento 'what_data' debe ser 'trn' para los datos de training, 'val' para los datos de validación, o 'tst' para los datos de testing.")
        return self._success_rate_from_data(x_data, y_data)
    # El siguiente método hace que se entrene.
    def train(self) -> None:
        
        for i in range(0, self.max_iterations):
            for j in range(0, self.trn_rows()):
                y_real = self.y_trn[j]
                y_calculado = self.output(self.x_trn[j]) # Paso 1) Consigue el y(n)
                self.update_w(self.x_trn[j], y_calculado, y_real)   # Paso 2) Actualiza el w(n) a w(n+1)
            success_rate = self.success_rate("val")
            if success_rate >= self.success_rate_criteria:
                print("Se ha alcanzado un criterio de éxito de " + str(success_rate*100) + "%, que es mayor o igual a " + str(self.success_rate_criteria*100) + "% por lo tanto se frena el entrenamiento en la iteración número " + str(i))
                print("🚀🚀🚀 🔥🔥🔥 😎😎😎")
                return
        print("Se ha alcanzado el número máximo de iteraciones y no se ha alcanzado el criterio de éxito esperado 😞")
        print("El criterio obtenido fue de " + str(success_rate*100) + "% y el esperado es de " + str(self.success_rate_criteria*100) + "%")
    def test_data(self) -> None:
        success_rate = self.success_rate("tst")
        print("El criterio de éxito en los patrones de prueba es: " + str(success_rate*100) + "% !!!")
        if (success_rate >= self.success_rate_criteria):
            print("Superó el " + str(self.success_rate_criteria*100) + "%, habéis pasado la prueba 😎")
        else:
            print("No superó el " + str(self.success_rate_criteria*100) + "%, no habéis pasado la prueba 😞")
