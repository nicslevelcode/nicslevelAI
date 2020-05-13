# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

'''documentation keywords'''
# chng: changes might be required depending on your code

'''import required libraries'''
# Basic libraries
import pandas as pd
import numpy as np
import datetime as d
# import seaborn as sn
import matplotlib.pyplot as plt

file_path = 'C:/Users/nicholasleongzw/Documents/GitHub/AI4I-Repo/SUP2/train.csv' # chng 
df = pd.read_csv(file_path,index_col=0) # chng 

def EDA(df=df,mc=False,mr=False):
    if mc:
        pd.set_option('max_columns', None)
    if mr:
        pd.set_option('max_rows', None)     
    print('First few rows...')
    print(df.head())
    print('Last few rows...')
    print(df.tail())
    print('\n')
    print(df.describe())
    print(df.info())
    '''other useful functions'''
    # print(df.nunique())
    # non_numeric_df = df.select_dtypes(include=['object']) # chng 
    # numeric_df = df.select_dtypes(include=['int64','float64']) # chng 
    # for i in df:
        # print(df[i].value_counts())

# EDA() # chng

list = ['Alley','PoolQC','MiscFeature','MSSubClass']
for i in list:
    print(df[i].value_counts())  

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
# sn.heatmap(corr, mask=mask, center=0, linewidths=1, fmt='.1f', annot=True, xticklabels=True, yticklabels=True)