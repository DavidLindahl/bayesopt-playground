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



def plot_metric_over_iterations(df, metric, title, ylabel):
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
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
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

def plot_hyperparameter_evolution(df, hyperparams):
    """
    Create subplots to visualize how individual hyperparameters evolve over iterations.
    
    Parameters:
      - df: DataFrame containing:
            - 'iteration': The iteration number.
            - 'acq_func': Acquisition function identifier.
            - Plus one column per hyperparameter in the hyperparams list.
      - hyperparams: List of hyperparameter names (strings) to plot
                     (e.g., ['learning_rate', 'dropout_rate']).
    """
    num_params = len(hyperparams)
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 4 * num_params), sharex=True)
    
    # If there's only one hyperparameter, axes is not a list.
    if num_params == 1:
        axes = [axes]

    for i, hp in enumerate(hyperparams):
        sns.lineplot(data=df, x="iteration", y=hp, hue="acq_func", marker="o", ax=axes[i])
        axes[i].set_title(f"Evolution of {hp} over Iterations")
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel(hp)
        axes[i].legend(title="Acquisition Function")
    
    plt.tight_layout()
    plt.show()

def plot_acquisition_function_values(df):
    """
    Plot the acquisition function values over iterations for each acquisition function.
    
    Parameters:
      - df: DataFrame containing:
          - 'iteration': The iteration number.
          - 'acq_func': The acquisition function used (e.g., 'EI', 'PI', 'LCB').
          - 'acq_value': The value of the acquisition function at that iteration.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="iteration", y="acq_value", hue="acq_func", marker="o")
    plt.title("Acquisition Function Values over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Acquisition Function Value")
    plt.legend(title="Acquisition Function")
    plt.tight_layout()
    plt.show()


### EXAMPLES ###
def main():
    data_file = "/Users/m.brochlips/Programering/AI/Projects/bayesian-optimization/BO_results.csv"
    df = load_data(data_file)
    
    # Plot accuracy over iterations for each acquisition function.
    plot_metric_over_iterations(
        df,
        metric="accuracy",
        title="Accuracy over Iterations for Different Acquisition Functions",
        ylabel="Accuracy (%)"
    )
    
    # # Plot loss over iterations.
    # plot_metric_over_iterations(
    #     df,
    #     metric="loss",
    #     title="Loss over Iterations for Different Acquisition Functions",
    #     ylabel="Loss"
    # )
    
    # Plot kernel_size_1 over iterations.
    plot_metric_over_iterations(
        df,
        metric="kernel_size_1",
        title="kernel_size_1 over Iterations for Different Acquisition Functions",
        ylabel="kernel_size_1 (e.g., number of parameters)"
    )
    
    # Scatter plot for model size vs. accuracy.
    plot_maxpool_size_vs_accuracy(df)
    
    # Plot acquisition function values over iterations.
    plot_acquisition_function_values(df)

if __name__ == "__main__":
    main()
