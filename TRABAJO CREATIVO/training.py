from features import Features
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from red import CNN
#-----------------------------------ENTRENAMIENTO-----------------------------------#

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
train_data = TensorDataset(x_trn_padded_torch, y_trn_torch) #Un dataset permite guardar datos para posteriormente entrenar/testear un modelo
test_data = TensorDataset(x_tst_padded_torch, y_tst_torch)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True) #un dataloader es como un contenedor de un dataset, tiene batches, iteradores, etc
test_loader = DataLoader(test_data, batch_size=128)

model = CNN(embedding_matrix_torch).to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Preguntarle a di persia que prefiere
#optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3) #
target_acc = 0.95   # detener si llegamos al 95%
tol = 0.005
last_acc = 0
for epoch in range(50):
    model.train()
    total_loss = 0
    aciertos = 0
    total = 0
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)              
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Convertimos lo que tira logits a 0 o 1 usando una sigmoide truncada
        preds = torch.sigmoid(outputs) >= 0.5
        aciertos += (preds.float() == y_batch).sum().item()
        total += y_batch.size(0) #cantidad de filas del batch

    acc = aciertos / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # criterio de parada por accuracy
    if acc >= target_acc:
        print("Accuracy requerida alcanzada.")
        break
    if abs(acc-last_acc)<tol:
        print("Detenido por no mejora en la Accuracy")
        break
    else:
        last_acc=acc

# Guardamos el modelo
torch.save(model.state_dict(), r'TRABAJO CREATIVO/model.model')