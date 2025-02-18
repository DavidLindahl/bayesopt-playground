from skopt.space import Integer, Categorical
from torch import nn
import model as CNNmodel
from BO import BO

# Set seeds
np.random.seed(0)
torch.manual_seed(0)




# CNNmodel hyperparameters for optimization
dimensions = [
    Integer(1, 10, name='n_layers'),
    Integer(32, 512, name='n_units'),
    Categorical(['relu', 'sigmoid', 'tanh'], name='activation')
]

# Optimizer parameters
optimizer_params = {
    'n_calls': 10,
    'n_initial_points': 5,
    'initial_point_generator': 'sobol'
    'aqcuisition': 'gp_hedge',
    'n_points': 1000,
    'verbose': True
}

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

OptimizeResult = BO(
    CNNmodel, 
    dimensions, 
    dataloader, 
    val_dataloader, 
    optimizer_params,
    ):