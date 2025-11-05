from features import Features
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#Clase RED CONVOLUCIONAL
class CNN(nn.Module):
    def __init__(self, features, embedding_matrix, num_filters=128, kernel_size=5):
        super().__init__()
        self.num_filters=num_filters
        self.kernel_size=kernel_size
        max_len = len(features.X_trn_padded[0, :])
        vocab_size, embedding_dim = embedding_matrix.shape
        # El siguiente embedding se utiliza como paso del forward, para pasar de un vector de tokens a un vector denso.
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,       # No entrenamos embeddings
            padding_idx=0      # El 0 es el padding
        )
        # Capa Convolucional, nos entrega un mapa de caracteristicas (dim=128) que se usan como filtros
        self.Conv1=nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5,stride=1)
        self.Conv2=nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3,stride=1)
        # Capa de POOLING, Se reduce el mapa de caracteristicas mediante muestreo (en este caso de valor maximo).
        # Kernel: Tamaño de ventana.
        # Stride: Paso para tomar la ventana.
        # Por cada ventana toma el máximo.
        self.Pool1d = nn.MaxPool1d(kernel_size=2,stride=2)  
        self.Pool1d2 = nn.MaxPool1d(kernel_size=2,stride=2)  
        # Capa de DROPOUT, convierto a 0 elementos randoms de los inputs con probabilidad p
        # Hinton comprobo que esto ayuda a evitar el sobreentrenamiento/overfitting.
        # Es para que la red no dependa de neuronas específicas. Apagando algunas neuronas, obligo a las otras a compensar
        self.Dropout=nn.Dropout(p=0.5)
        self.Dropout2=nn.Dropout(p=0.5)
        # Capa de Flatten, transforma todo a un tensor de 1 dimension
        self.flatten=nn.Flatten()
        
        # Capa Dense, Funcion lineal rectificadora , negativos a 0 
        self.ReLU=nn.ReLU()
        self.ReLU2=nn.ReLU()
        # Output sigmoide de toda la vida (Va de 0 a 1)
        self.Sigmoid=nn.Sigmoid()

        # Calcular dinámicamente el tamaño del feature map resultante
        with torch.no_grad():
            sample_input = torch.zeros(1, max_len, embedding_matrix.shape[1])  # (batch=1, seq_len, emb_dim)
            sample_input = sample_input.permute(0, 2, 1)  # (1, emb_dim, seq_len)
            x = self.Conv1(sample_input)
            x = self.ReLU(x)
            x = self.Pool1d(x)
            x = self.Conv2(x)
            x = self.ReLU(x)
            x = self.Pool1d2(x)
            feature_dim = x.numel()  # cantidad total de elementos después del flatten

        self.fc = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        # x = self.embedding(x)  ->   (batch, seq_len) -> (batch, seq_len, emb_dim)   
        # x = x.permute(0, 2, 1) ->   Reorganiza: (batch, emb_dim, seq_len)
        x = x.permute(0, 2, 1) # Se permuta simplemente para adaptar los datos de 'self.embedding(x)' al input esperado de la capa convolucional.      
        x = self.Conv1(x)
        x = self.ReLU(x)
        x = self.Pool1d(x)
        x = self.Dropout(x)
        x = self.Conv2(x)
        x = self.ReLU2(x)
        x = self.Pool1d2(x)
        x = self.Dropout2(x)
        x = self.flatten(x)

        x = self.fc(x)
        #x = self.Sigmoid(x)
        return x
# No usamos sigmoide al final porque el criterio ya tiene una sigmoide incluida.
