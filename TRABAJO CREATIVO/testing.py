from features import Features
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import F1Score
from features import Features

#Preparacion de los datos para training
def test(f: Features, modelObj, model_path: str):

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

    model = modelObj(f, embedding_matrix_torch).to(device)
    #r'TRABAJO CREATIVO/model.model'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    #-----------------------------------EVALUACION-----------------------------------#
    criterion = nn.BCEWithLogitsLoss()
    model.eval() # Capas como el Dropout (Útil para entrenar) no se toman en cuenta.
    total_loss = 0
    aciertos = 0
    total = 0

    preds_l = []
    y_l = []

    with torch.no_grad():   
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)              
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # Convertimos lo que tira logits a 0 o 1 usando una sigmoide truncada
            preds = torch.sigmoid(outputs) >= 0.5
            preds_l = preds_l + preds.tolist()
            y_l = y_l + y_batch.tolist()
            aciertos += (preds.float() == y_batch).sum().item()
            total += y_batch.size(0) #cantidad de filas del batch
            
    #Medida de accuracy
    acc = aciertos / total
    avg_loss = total_loss / len(test_loader)
    print(f"TEST : , Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")


    # Medida de F1
    f1_binary = F1Score(task="binary")
    score_binary = f1_binary(torch.tensor(preds_l), torch.tensor(y_l))
    print("F1 Score: " + str(score_binary))