import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from yellowbrick.regressor import ResidualsPlot, PredictionError

from scipy.stats import yeojohnson

custom_params = {
    "axes.spines.right": False, 
    "axes.spines.top": False
    }

def plot_perc_missing(
    df:pd.DataFrame, 
    dtype_include:list=None, 
    title:str=r'% of Missing Values',
    figsize:tuple=(6,12),
    *args, **kwargs) -> None:
    
    if dtype_include is None:
        dtype_include=df.dtypes.unique()
    
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xticks(range(0,101,20))
    plt.xlabel('% Missing Values')
    plt.axvline(x=50, color='g', linestyle='--')
    plt.axvline(x=90, color='r', linestyle='--')
    df\
        .select_dtypes(include=dtype_include)\
        .apply(lambda x: x.isnull().mean()*100)\
        .sort_values(ascending=True)\
        .plot(kind='barh', *args, **kwargs)
    plt.show()


def plot_feature_correlations(
    df:pd.DataFrame, 
    col_target:str, 
    title:str=None,
    figsize:tuple=(6,12),
    threshold:float=None,
    *args, **kwargs) -> None:
    
    plt.figure(figsize=figsize)
    if title is None:
        title=f'Correlation with {col_target}'
    plt.title(title)
    plt.xlabel(f'Correlation')
    plt.xlim((0.0,1.0))
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--')
    df\
        .corr()[col_target]\
        .abs()\
        .sort_values(ascending=True)\
        .drop(col_target)\
        .plot(kind='barh', *args, **kwargs)
    plt.show()

def histogram_target_missingness_relationship(
    df:pd.DataFrame,
    col_target:str, 
    col_var:str, 
    *args,
    **kwargs):
    
    df_tmp = df.copy()
    
    col_na = col_var + '_missing'
    df_tmp[col_na] = df_tmp[col_var].isnull()

    sns.set_theme(
        style="ticks", 
        palette=sns.color_palette("Set1"), 
        rc=custom_params)
        
    sns.histplot(
        x=df_tmp[col_target], 
        hue=df_tmp[col_na],
        *args, **kwargs
        )
    
    plt.title(f'Distribution of \'{col_target}\' \nbased on missingness of \'{col_na}\' values')
    plt.ylabel('COUNT')
    plt.xlabel(col_target.replace('_', ' ').upper())
    plt.show()

def boxplot_target_missingness_relationship(
    df:pd.DataFrame,
    col_target:str, 
    col_var:str, 
    *args,
    **kwargs):
    
    df_tmp = df.copy()

    col_na = col_var + '_missing'
    df_tmp[col_na] = df_tmp[col_var].isnull()
    
    sns.set_theme(
        style="ticks", 
        palette=sns.color_palette("Set1"), 
        rc=custom_params)
       
    sns.boxplot(
        y=df_tmp[col_na].astype(str), 
        x=df_tmp[col_target], 
        *args, **kwargs)
    plt.title(f'Distribution of \'{col_target}\' \nbased on missingness of \'{col_na}\' values')
    plt.ylabel(col_na.replace('_', ' ').upper())
    plt.xlabel(col_target.replace('_', ' ').upper())
    plt.show()
    

def boxplot_histogram_missingness_relationship(
    df:pd.DataFrame,
    col_target:str, 
    col_var:str, 
    add_yeojohnson:bool=False,
    add_kde:bool=False,
    multiple:str='stack',
    *args,
    **kwargs
):
    
    warnings.filterwarnings("ignore", category=UserWarning)

    if add_yeojohnson:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(18,5))
    else:
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(12,5))

    df_tmp = df.copy()

    col_na = col_var + '_missing'
    df_tmp[col_na] = df_tmp[col_var].isnull()

    sns.set_theme(
        style="ticks", 
        palette=sns.color_palette("Set1"), 
        rc=custom_params)

    sns.boxplot(
        y=df_tmp[col_na].astype(str), 
        x=df_tmp[col_target], 
        ax=ax1
        )
    ax1.set_title(f'Boxplot of \'{col_target}\'')

    sns.histplot(
        x=df_tmp[col_target], 
        hue=df_tmp[col_na],
        kde=add_kde,
        ax=ax2,
        multiple=multiple,
        )
    ax2.set_title(f'Distribution of \'{col_target}\'')
    ax2.set_xlabel(col_target)
    ax2.set_ylabel('Count')

    if add_yeojohnson:
        sns.histplot(
            x=yeojohnson(df_tmp[col_target])[0], 
            hue=df_tmp[col_na],
            kde=add_kde,
            ax=ax3
            )
        ax3.set_title(f'Distribution of \'{col_target}\' with Yeo-Johnson transformation')
        ax3.set_xlabel(f'Yeo-Johnson({col_target})')
        ax3.set_ylabel('Count')
        
    plt.rc('legend', loc='upper right')
    plt.suptitle(f'Distribution of \'{col_target}\' based on missingness of \'{col_var}\' values')
    plt.show(*args, **kwargs)
    
    
def plot_correlation_heatmap(
    df:pd.DataFrame, 
    cols:list=None, 
    title:str=r'Correlation Heatmap',
    figsize:tuple=(14,12),
    *args, **kwargs) -> None:
    
    if cols is None:
        cols=df.columns
    
    plt.figure(figsize=figsize)
    plt.title(title)
    sns.heatmap(
        data=df[cols].corr(),
        cmap='RdBu_r',
        *args, **kwargs
    )
    plt.show()
    

def plot_percentage_unique(
    df:pd.DataFrame, 
    dtype_include:list=None, 
    title:str='% of Unique Values',
    figsize:tuple=(6,12),
    *args, **kwargs) -> None:
    
    if dtype_include is None:
        dtype_include=df.dtypes.unique()
    
    plt.figure(figsize=figsize)
    plt.title(title)
    df\
        .select_dtypes(include=dtype_include)\
        .apply(lambda x: len(x.unique())/len(x)*100)\
        .sort_values(ascending=True)\
        .plot(kind='barh', *args, **kwargs)
    plt.xlim((0,100))
    plt.xlabel('% of Unique Values')
    plt.show()
    

def plot_count_unique(
    df:pd.DataFrame, 
    dtype_include:list=None, 
    title:str='Count of Unique Values',
    figsize:tuple=(6,12),
    *args, **kwargs) -> None:
    
    if dtype_include is None:
        dtype_include=df.dtypes.unique()
    
    plt.figure(figsize=figsize)
    plt.title(title)
    df\
        .select_dtypes(include=dtype_include)\
        .apply(lambda x: len(x.unique()))\
        .sort_values(ascending=True)\
        .plot(kind='barh', *args, **kwargs)
    plt.xlabel('Count of Unique Values')
    plt.show()
    
def plot_pred_obs(model, x_train, y_train, 
                  x_test=None, y_test=None, 
                  name:str=None, 
                  save_path: str = None, 
                  show:bool=False,
                  *args, **kwargs
                  ):

    visualizer = PredictionError(model, *args, **kwargs)
    visualizer.fit(x_train, y_train)
    
    if (x_test is not None) and (y_test is not None):
        visualizer.score(x_test, y_test)
        # source_data = '- Test Set'
    else:
        visualizer.score(x_train, y_train)
        # source_data = '- Training Set'

    if name is None:
        plt.suptitle(f'Predicted vs Observed Values')
    else:
        plt.suptitle(f'{name}: Predicted vs Observed Values')
    
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        visualizer.show()
    visualizer.finalize()

def plot_residuals(model, x_train, y_train, 
                  x_test=None, y_test=None, 
                  name:str=None, 
                  save_path: str = None, 
                  show:bool=False,
                  *args, **kwargs
                  ):

    visualizer = ResidualsPlot(model, *args, **kwargs)
    visualizer.fit(x_train, y_train)
    
    if (x_test is not None) and (y_test is not None):
        visualizer.score(x_test, y_test)

    if name is None:
        plt.suptitle(f'Residuals')
    else:
        plt.suptitle(f'{name}: Residuals')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        visualizer.show()
    visualizer.finalize()

def plot_stacked_bar(df, labels, colors, title, subtitle, xlabel):
    fields = df.columns.tolist()
    
    # figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 20))
    
    # plot bars
    left = len(df) * [0]
    for idx, name in enumerate(fields):
        plt.barh(df.index, df[name], left = left, color=colors[idx])
        left = left + df[name]
    
    # title and subtitle
    plt.title(title, loc='left', fontsize=14)
    plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle, fontsize=12)
    # legend
    plt.legend(labels, bbox_to_anchor=([0.58, 1, 0, 0]), ncol=4, frameon=False)
    
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.xlabel(xlabel=xlabel)
    # adjust limits and draw grid lines
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    
    plt.show()