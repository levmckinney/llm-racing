import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
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
        A Seaborn figure containing a scatter plot for each model comparing
        the number of tokens generated to the time taken to generate them.
    """
    plt.clf()
    # Create a new Seaborn figure
    fig = sns.scatterplot(data=data_frame, x='tokens', y='time', hue='model_name')

    # Add labels and title to the plot
    fig.set(xlabel='Tokens Generated', ylabel='Time Taken (s)', title='Tokens Generated vs Time Taken for Each Model')

    # Return the figure
    return fig.figure

def plot_results(data_frame: pd.DataFrame, ci: float = 0.95, include_legend: bool = True, include_x_ticks: bool = True) -> plt.Figure:
    """
    Plot the results of a time trial assuming some startup costs.

    Args:
        data_frame: DataFrame where each row is a sample and the columns are:
            - `model_name`: Name of the model.
            - `tokens`: Number of tokens generated.
            - `time`: Time taken to generate tokens.
        ci: Confidence interval to plot.

    Returns:
        A Seaborn figure where the x-axis is the model name and the y-axis is 
        the number of tokens generated per second with a confidence interval.
    """
    plt.clf()
    grouped_data = data_frame.groupby('model_name')

    model_names = []
    slopes = []
    lower_bounds = []
    upper_bounds = []

    # Iterate through each group (unique model_name) in the grouped data
    for model_name, group in grouped_data:
        # Set X to be the 'time' column and y to be the 'tokens' column
        X = group['time']
        y = group['tokens']

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


    # Create a new DataFrame from the extracted data
    results_df = pd.DataFrame({'Model Name': model_names, 'Tokens per Second': slopes,
                               'Lower Bound': lower_bounds, 'Upper Bound': upper_bounds})


    results_df.sort_values(by='Tokens per Second', inplace=True)
    results_df.index = range(len(results_df))
    print(results_df)
    # create a color palette the length of the dataframe
    colors = sns.color_palette('husl', n_colors=len(results_df))


    # Create a new Seaborn barplot
    fig = sns.barplot(
        data=results_df,
        x='Model Name',
        y='Tokens per Second', 
        hue='Model Name',
        dodge=False,
        palette=colors,
    )

    for i, row in enumerate(results_df.itertuples()):
        fig.errorbar(i, row._2, yerr=[[row._2 - row._3], [row._4 - row._2]], capsize=10, fmt='none', color='black')

    if include_x_ticks:
        plt.xticks(rotation=90)
    else:
        fig.set(xticklabels=[])
        fig.tick_params(bottom=False)

    if not include_legend:
        fig.legend_.remove()

    return fig.figure