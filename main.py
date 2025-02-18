from skopt.space import Integer, Categorical
import torch
from model.CNN_model import CNN
from BO import BO
from data.data_loader import load_MNIST
import numpy as np

# Set seeds
np.random.seed(0)
torch.manual_seed(0)


# CNNmodel hyperparameters for optimization
dimensions = [
    Integer(8, 48, name="conv_nodes_1"),
    Integer(8, 48, name="conv_nodes_2"),
    Categorical([3, 5], name="kernel_size_1"),
    Categorical([3, 5], name="kernel_size_2"),
    Integer(2, 4, name="maxpool_size"),
    Categorical([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], name="dropout_rate"),
    Integer(32, 512, name="fc_nodes"),
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



OptimizeResult = BO(
    Model_class=CNN,
    dimensions = dimensions,
    train_dataloader = train_loader,
    val_dataloader = val_loader,
    optimizer_params,
    ):
