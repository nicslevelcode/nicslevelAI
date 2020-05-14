# Basic libraries
import pandas as pd
import numpy as np
import datetime as d
import seaborn as sn
import matplotlib.pyplot as plt

test_path = 'C:/Users/nicho/Documents/GitHub/AnacondaML/SUP2/test.csv'
test_df = pd.read_csv(test_path,index_col=0) # chng 

print(test_df.head())