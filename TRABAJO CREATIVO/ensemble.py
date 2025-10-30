# pip install torchensemble

# Me hice un copy-paste medio extraño, despues lo termino bien :p

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchensemble import VotingClassifier  # voting is a classic ensemble strategy

# Define your base deep learning model (e.g., a simple MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
base_estimator = SimpleMLP(input_size=10, num_classes=2)

# Load your data (example placeholders)
train_loader = DataLoader(torch.randn(100, 10), batch_size=16) # Example: 100 samples, 10 features
test_loader = DataLoader(torch.randn(50, 10), batch_size=16)  # Example: 50 samples, 10 features

# Define the ensemble
ensemble = VotingClassifier(
    estimator=base_estimator,               # here is your deep learning model
    n_estimators=10,                        # number of base estimators
)
# Set the criterion
criterion = nn.CrossEntropyLoss()           # training objective
ensemble.set_criterion(criterion)

epochs = 10
learning_rate = 1e-3
weight_decay = 0.01

# Set the optimizer
ensemble.set_optimizer(
    "Adam",                                 # type of parameter optimizer
    lr=learning_rate,                       # learning rate of parameter optimizer
    weight_decay=weight_decay,              # weight decay of parameter optimizer
)

# Set the learning rate scheduler
ensemble.set_scheduler(
    "CosineAnnealingLR",                    # type of learning rate scheduler
    T_max=epochs,                           # additional arguments on the scheduler
)

# Train the ensemble
ensemble.fit(
    train_loader,
    epochs=epochs,                          # number of training epochs
)

# Evaluate the ensemble
acc = ensemble.predict(test_loader)         # testing accuracy