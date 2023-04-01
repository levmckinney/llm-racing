import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm



def scatter_plots(data_frame: pd.DataFrame) -> plt.Figure:
    """
    Create scatter plots for each model.

    Args:
        data_frame: DataFrame where each row is a sample and the columns are:
            - `model_name`: Name of the model.
            - `tokens`: Number of tokens generated.
            - `time`: Time taken to generate tokens.

    Returns:
        A matplotlib figure containing a scatter plot for each model comparing
        the number of tokens generated to the time taken to generate them.
    """
    # Create a new matplotlib figure
    fig, ax = plt.subplots()

    # Get the unique model names from the data_frame
    model_names = data_frame["model_name"].unique()

    # Iterate through the unique model names
    for model_name in model_names:
        # Extract the data for the current model
        model_data = data_frame[data_frame["model_name"] == model_name]

        # Create a scatter plot for the current model
        ax.scatter(model_data["tokens"], model_data["time"], label=model_name)

    # Add labels, title, and legend to the plot
    ax.set_xlabel("Tokens Generated")
    ax.set_ylabel("Time Taken (s)")
    ax.set_title("Tokens Generated vs Time Taken for Each Model")
    ax.legend()

    # Return the figure
    return fig

def plot_results(data_frame: pd.DataFrame, ci: float = 0.95) -> plt.Figure:
    """
    Plot the results of a time trial assuming some startup costs.

    Args:
        data_frame: DataFrame where each row is a sample and the columns are:
            - `model_name`: Name of the model.
            - `tokens`: Number of tokens generated.
            - `time`: Time taken to generate tokens.
        ci: Confidence interval to plot.

    Returns:
        A matplotlib figure where the x-axis is the model name and the y-axis is 
        the number of tokens generated per second with a confidence interval.
    """
    grouped_data = data_frame.groupby('model_name')

    model_names = []
    slopes = []
    lower_bounds = []
    upper_bounds = []

    # Iterate through each group (unique model_name) in the grouped data
    for model_name, group in grouped_data:
        # Set X to be the 'tokens' column and y to be the 'time' column
        X = group['tokens']
        y = group['time']

        # Add a constant term to the X matrix for the linear regression (for the intercept)
        X = sm.add_constant(X)

        # Perform the linear regression using the statsmodels library
        model = sm.OLS(y, X).fit()

        # Compute the confidence interval for the model parameters (alpha is the significance level)
        conf_int = model.conf_int(alpha=(1 - ci))

        # Extract the slope (tokens per second) from the model parameters
        slope = model.params[1]

        # Extract the lower and upper bounds of the confidence interval for the slope
        lower_bound, upper_bound = conf_int.iloc[1]

        # Append the results to the corresponding lists
        model_names.append(model_name)
        slopes.append(slope)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Create a new plot
    fig, ax = plt.subplots()

    # Create the x values as integer indices for the bar plot
    x = np.arange(len(model_names))

    # Create a bar plot with error bars for the confidence intervals
    ax.bar(x, slopes, yerr=[np.array(slopes) - np.array(lower_bounds), np.array(upper_bounds) - np.array(slopes)], capsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)

    ax.set_xlabel('Model Name')
    ax.set_ylabel('Tokens Generated per Second')
    ax.set_title('Tokens per Second vs Model')

    return fig