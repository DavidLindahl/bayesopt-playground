from skopt.space import Integer, Categorical
import torch
import model as CNNmodel
from BO import BO
from data.data_loader import load_MNIST
import numpy as np

# Set seeds
np.random.seed(0)
torch.manual_seed(0)


# CNNmodel hyperparameters for optimization
dimensions = [
    Integer(1, 10, name="n_layers"),
    Integer(32, 512, name="n_units"),
    Categorical(["relu", "sigmoid", "tanh"], name="activation"),
]

# Optimizer parameters
optimizer_params = {
    "n_calls": 10,
    "n_initial_points": 5,
    "initial_point_generator": "sobol",
    "aqcuisition": "gp_hedge",
    "n_points": 1000,
    "verbose": True,
}

# Data loader parameters
data_loader_params = {
    "train_size": 50000,
    "test_size": 10000,
    "val_size": 10000,
    "batch_size": 32,
    "shuffle": True,
}

train_loader, val_loader, test_loader = load_MNIST(**data_loader_params)

# # Check one batch from the training loader
# for images, labels in train_loader:
#     print("Train batch:", images.shape, labels.shape)
#     break

# if val_loader is not None:
#     for images, labels in val_loader:
#         print("Validation batch:", images.shape, labels.shape)
#         break

# for images, labels in test_loader:
#     print("Test batch:", images.shape, labels.shape)
#     break

OptimizeResult = BO(
    CNNmodel,
    dimensions,
    train_loader,
    val_dataloader,
    optimizer_params,
    ):
