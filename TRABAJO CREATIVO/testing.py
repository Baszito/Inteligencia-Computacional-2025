from features import Features
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from red import CNN

#Features Engineering
f = Features() 

#Configuracion, para mandar todo el modelo a la GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# max_len es la cantidad de tokens que tiene cada secuencia con padding incluido.
max_len = len(f.X_trn_padded[0, :])

#Transformar los datos del Features Engineering al formato de TORCH (Tensores)
x_trn_padded_torch = torch.from_numpy(f.X_trn_padded)
x_tst_padded_torch = torch.from_numpy(f.X_tst_padded)

y_trn_torch = torch.from_numpy(f.y_trn)
y_tst_torch = torch.from_numpy(f.y_tst)
embedding_matrix_torch = torch.tensor(f.matriz_embedding, dtype=torch.float32)

#Casteo a flotantes, para que despues no moleste en el criterion
y_trn_torch = y_trn_torch.float()
y_tst_torch = y_tst_torch.float()

# Mover los tensores al device de TORCH
if torch.accelerator.is_available():
    x_trn_padded_torch = x_trn_padded_torch.to(device)
    x_tst_padded_torch = x_tst_padded_torch.to(device)
    y_trn_torch = y_trn_torch.to(device)
    y_tst_torch = y_tst_torch.to(device)

# Crear datasets y dataloaders
test_data = TensorDataset(x_tst_padded_torch, y_tst_torch)

test_loader = DataLoader(test_data, batch_size=128)

model = CNN(embedding_matrix_torch).to(device)
model.load_state_dict(torch.load(r'TRABAJO CREATIVO/model.model', weights_only=True))
model.eval()

#-----------------------------------EVALUACION-----------------------------------#

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Preguntarle a di persia que prefiere
#optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3) #
target_acc = 0.95   # detener si llegamos al 95%
tol = 0.005
last_acc = 0

model.eval() # Capas como el Dropout (Útil para entrenar) no se toman en cuenta.
total_loss = 0
aciertos = 0
total = 0

with torch.no_grad():   
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)              
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()

        # Convertimos lo que tira logits a 0 o 1 usando una sigmoide truncada
        preds = torch.sigmoid(outputs) >= 0.5
        aciertos += (preds.float() == y_batch).sum().item()
        total += y_batch.size(0) #cantidad de filas del batch

acc = aciertos / total
avg_loss = total_loss / len(test_loader)
print(f"TEST : , Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")