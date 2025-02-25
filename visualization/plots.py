import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Seaborn style for nicer plots
sns.set_theme(style="whitegrid")

def load_data(file_path):
    """
    Load results data from a CSV file.
    
    The CSV is expected to have at least the following columns:
      - iteration: the iteration number in the optimization loop.
      - acq_func: name of the acquisition function used.
      - acq_value: the value of the acquisition function at that iteration.
      - accuracy: the validation or test accuracy achieved.
      - loss: the validation or test loss achieved.
      - model_size: a metric representing the model size (e.g., number of parameters).
      
    Additional hyperparameter columns may also be present.
    """
    return pd.read_csv(file_path)



def plot_metric_over_iterations(df, metric):
    """
    Plot the given metric over iterations for each acquisition function.
    
    Parameters:
      - df: DataFrame with columns including 'iteration', 'acq_func', and the metric to plot.
      - metric: The name of the metric column to visualize.
      - title: Title for the plot.
      - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="iteration", y=metric, hue="acq_func", marker="o")
    plt.title(f'{metric} over iterations')
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.legend(title="Acquisition Function")
    plt.tight_layout()
    plt.show()

def plot_maxpool_size_vs_accuracy(df):
    """
    Create a scatter plot showing the trade-off between model size and accuracy.
    
    Parameters:
      - df: DataFrame containing at least:
          - 'model_size': Model size metric (e.g., number of parameters).
          - 'accuracy': Accuracy value.
          - 'acq_func': Acquisition function identifier.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="maxpool_size", y="accuracy", hue="acq_func",
                    palette="deep", s=100, alpha=0.8)
    plt.title("Maxpool size vs. Accuracy")
    plt.xlabel("Maxpool size (e.g., number of parameters)")
    plt.ylabel("Accuracy")
    plt.legend(title="Acquisition Function")
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_evolution(df, acq_func):
    """
    Visualize the evolution of grouped hyperparameters over iterations for a single acquisition function.
    
    Parameters:
      - df: DataFrame containing:
            - 'iteration': The iteration number.
            - 'acq_func': Acquisition function identifier.
            - Various hyperparameter columns.
      - acq_func: The acquisition function to filter for plotting.
    """
    df_filtered = df[df['acq_func'] == acq_func]
    
    param_groups = {
        "Kernel & Pooling": ['kernel_size_1', 'kernel_size_2', 'maxpool_size', 'dropout_rate'],
        "Convolutional Nodes": ['conv_nodes_1', 'conv_nodes_2']
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    for (title, params), ax in zip(param_groups.items(), axes):
        for hp in params:
            sns.lineplot(data=df_filtered, x="iteration", y=hp, marker="o", label=hp, ax=ax)
        ax.set_title(f"{title} Evolution for {acq_func}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hyperparameter Value")
        ax.legend(title="Hyperparameters")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()