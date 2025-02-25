from plots import*
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
