import os
import pickle as pkl
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def reset_index_df (df:pd.DataFrame)->pd.DataFrame:
    """
    Function to reset dataframe with epoch as column

    Args:
        df [pandas dataframe]: loss and accuracy dataframe with epoch as index

    Returns:
        new_df [pandas dataframe]: dataframe with epoch reset to column
    """
    new_df = df.reset_index()
    new_df.rename(columns={'index':'epoch'},inplace=True)
    new_df['epoch'] = new_df['epoch'] + 1

    return new_df

def clean_metrics_df(loss_res:pd.DataFrame, acc_res:pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean metrics DataFrames by resetting the index, renaming columns, and adjusting epoch values.

    Args:
    - loss_res (pd.DataFrame): DataFrame containing loss metrics.
    - acc_res (pd.DataFrame): DataFrame containing accuracy metrics.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing cleaned loss DataFrame and cleaned accuracy DataFrame.
    """
    # Reset index and rename columns for loss DataFrame
    loss_df = loss_res.reset_index()
    loss_df.rename(columns={'index':'epoch'},inplace=True)
    loss_df['epoch'] = loss_df['epoch'] + 1

    # Reset index and rename columns for accuracy DataFrame
    acc_df = acc_res.reset_index()
    acc_df.rename(columns={'index':'epoch'},inplace=True)
    acc_df['epoch'] = acc_df['epoch'] + 1

    return loss_df, acc_df

def load_results(run_id:str)->Tuple[pd.DataFrame, pd.DataFrame]:
    with open (os.path.join(f"../src/saved_model/{run_id}", 'model_train_results.pkl'), 'rb') as f:
        results = pkl.load(f)
        loss_res = pd.DataFrame(results[0])
        acc_res = pd.DataFrame(results[1])

    return loss_res, acc_res

def plot_train_results (loss_df:pd.DataFrame, acc_df:pd.DataFrame, dataset:str, title:bool=False, xlabel:str="", ylabel:bool=False, savefig:bool=False):
    """
    Function to plot 2 subplots of loss and accuracy subplots

    Args:
        loss_df [pandas dataframe]
        acc_df [pandas dataframe]
        dataset [str]: dataset name
        title [bool]: Optional, default is false; Insert title if true
        xlabel [str]: Optional, defautl is empty string. Rename xlabel.
        ylabel [bool]: Optional, defautl is false. Insert ylabel if true
    """

    fig, ax = plt.subplots(2,1, figsize=(7,10))
    sns.lineplot(data=loss_df.melt(id_vars='epoch'),x='epoch',y='value',hue='variable',ax=ax[0])

    sns.lineplot(data=acc_df.melt(id_vars='epoch'),x='epoch',y='value',hue='variable', ax=ax[1])

    if xlabel:
        ax[0].set_xlabel(f"{xlabel}")
        ax[1].set_xlabel(f"{xlabel}")

    if ylabel:
        ax[0].set_ylabel("Loss Values")
        ax[1].set_ylabel("Accuracy Values")

    if title:
        ax[0].set_title(f"Training, Validation and Test Loss Values for {dataset} dataset") 
        ax[1].set_title(f"Training, Validation and Test Accuracy Values for {dataset} dataset")
    
    if savefig:
        plt.savefig(f"../assets/results/full_train_results_{dataset}.png")

    return fig

def min_max_scaling(df: pd.DataFrame, column_name: str, feature_range: Tuple[float, float] = (0, 1)) -> pd.Series:
    """
    Perform min-max scaling on a column of DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to be scaled.
    - column_name (str): The name of the column to be scaled.
    - feature_range (Tuple[float, float]): The desired feature range after scaling. Default is (0, 1).

    Returns:
    - pd.Series: A Series containing the scaled values of the specified column.
    """
    # Calculate the minimum value in the specified column
    min_val = df[column_name].min()
    
    # Calculate the maximum value in the specified column
    max_val = df[column_name].max()
    
    # Perform min-max scaling on the specified column
    scaled_column = (df[column_name] - min_val) / (max_val - min_val)
    
    # Scale the values to the desired feature range
    scaled_column = scaled_column * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    # Return the scaled column
    return scaled_column

def load_all_results(run_id:str)-> Tuple[list, pd.DataFrame, pd.DataFrame]:
    
    with open (os.path.join(f"../src/saved_model/{run_id}", 'epoch_time.pkl'), 'rb') as f:
        epoch_time = pkl.load(f)

    with open (os.path.join(f"../src/saved_model/{run_id}", 'model_train_results.pkl'), 'rb') as f:
        results = pkl.load(f)
        loss_res = pd.DataFrame(results[0])
        acc_res = pd.DataFrame(results[1])

    return epoch_time, loss_res, acc_res

def concat_and_melt(epoch_time:list, loss_res:pd.DataFrame, acc_res:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    df = pd.concat([loss_res, acc_res, 
                    pd.DataFrame(epoch_time, columns=['epoch_time'])
                    ],axis=1) \
            .reset_index() \
            .rename(columns={'index':'epoch'})
    df['epoch'] = df['epoch'] + 1
    df_melted = df.melt(id_vars='epoch')

    return df, df_melted

def merge_dataset_runs(run_id_list: list[int], graph_combi_list: list[str]) -> pd.DataFrame:
    """
    Merge multiple datasets generated from different runs into a single DataFrame.

    Parameters:
    - run_id_list (list[int]): list of run IDs corresponding to different datasets.
    - graph_combi_list (list[str]): list of graph combinations for each run.

    Returns:
    - pd.DataFrame: Merged DataFrame containing data from all runs.
    """
    df_list = []  # list to store individual DataFrames from each run
    for run_id, graph_combi in zip(run_id_list, graph_combi_list):
        # Load data for the current run
        epoch_time, loss_res, acc_res = load_all_results(run_id=run_id)

        # Concatenate and melt data into a single DataFrame
        df, _ = concat_and_melt(epoch_time, loss_res, acc_res)

        # Add a new column 'graph_combi' to identify the graph combination
        df['graph_combi'] = graph_combi

        # Append the DataFrame to the list
        df_list.append(df)

    # Concatenate all DataFrames from different runs into a single DataFrame
    df = pd.concat(df_list)

    return df

def compute_mean (df:pd.DataFrame, col_name:str)->pd.DataFrame:
    """
    Function to calculate mean for each column over a wide dataframe of results and sort them in descending order. Epoch column is dropped in the process.

    Args:
        df [pandas dataframe]:  The input DataFrame.
        col_name [str]: name of column for mean values

    Returns
        new_df : dataframe of mean values for each graph combination
    """
    new_df = pd.DataFrame(df.drop(['epoch'],axis=1).mean()).reset_index().sort_values(by=0,ascending=False)
    new_df.columns = ['graph_combi',col_name]

    return new_df

def compute_epoch_mean (df:pd.DataFrame)->pd.DataFrame:
    """
    Function to calculate mean for epoch_time over a wide dataframe of results and sort them in descending order. E

    Args:
        df [pandas dataframe]:  The input DataFrame.

    Returns
        new_df : dataframe of mean values for each graph combination
    """
    new_df = df.groupby('graph_combi')[['epoch_time']].mean().reset_index().sort_values(by='epoch_time',ascending=False)

    return new_df

def plot_mean_epoch_time(df_list: list[pd.DataFrame], dataset_list: list[str]) -> pd.DataFrame:
    """
    Compute the mean epoch time for each dataset and plot them.

    Parameters:
    - df_list (list[pd.DataFrame]): list of DataFrames containing epoch time data for each dataset.
    - dataset_list (list[str]): list of dataset names.

    Returns:
    - pd.DataFrame: DataFrame containing mean epoch time for each dataset.
    """
    epoch_df_list = []  # list to store mean epoch time DataFrames for each dataset
    for df, dataset in zip(df_list, dataset_list):
        # Compute mean epoch time for the current dataset
        mean_epoch_df = compute_epoch_mean(df)
        
        # Add a new column 'dataset' to identify the dataset
        mean_epoch_df['dataset'] = dataset
        
        # Append the mean epoch time DataFrame to the list
        epoch_df_list.append(mean_epoch_df)

    # Concatenate all mean epoch time DataFrames into a single DataFrame
    epoch_df = pd.concat(epoch_df_list)

    return epoch_df