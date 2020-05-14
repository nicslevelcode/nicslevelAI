# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

'''documentation keywords'''
# chng: changes might be required depending on your code

'''import required libraries'''
# Basic libraries
import pandas as pd
import numpy as np
import datetime as d
import seaborn as sn
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
# Training and tuning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier

# Algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# Scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

file_path = 'C:/Users/nicho/Documents/GitHub/AnacondaML/SUP2/train.csv' # chng 
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

corr = df.corr()

abs_corr = abs(corr)
mask = abs_corr.loc[:,'SalePrice'] > 0.5
test = abs_corr[mask]
ft=test.index.tolist()
ft.remove('SalePrice')

test_path = 'C:/Users/nicho/Documents/GitHub/AnacondaML/SUP2/test.csv'
test_df = pd.read_csv(test_path,index_col=0,header=0) # chng
test_df=test_df[ft]
test_df.fillna(value=0.0, inplace=True)
print(test_df.info())

# mask = np.triu(np.ones_like(corr, dtype=bool))
# sn.heatmap(corr, mask=mask, center=0, linewidths=1, fmt='.1f', annot=True, xticklabels=True, yticklabels=True)
# plt.show()

def Main(R=True,VC=False):
    output_var_name = 'SalePrice' #chng
    output_var = df[output_var_name]
    df.drop(output_var_name,axis=1,inplace=True)
    
    '''custom preprocessing'''
    features = df[ft]
    # numerical_features = [key for key in dict(input_data.dtypes) if dict(input_data.dtypes)[key] in ['int64', 'float64']]  # chng
    # categorical_features = [key for key in dict(input_data.dtypes) if dict(input_data.dtypes)[key] in ['object']]  # chng
    # bin_features = [key for key in dict(input_data.dtypes) if dict(input_data.dtypes)[key] not in ['object']]  # chng
    
    numerical_features = features.select_dtypes(include=['int64','float64']).columns
    # categorical_features = features.select_dtypes(include=['object']).columns
    
    '''out of the box preprocessing'''
    preprocess = make_column_transformer(
        (make_pipeline(MinMaxScaler()), numerical_features), # chng
        # (make_pipeline(OneHotEncoder()), categorical_features), # chng
        # (make_pipeline(OneHotEncoder()), bin_features) # chng
        )

    ''' define your model(s)'''
    algo = KNeighborsRegressor(n_neighbors=5, weights='uniform') # chng
    model = make_pipeline(
        preprocess,
        algo 
        )
    
    if VC:
        algo_1 = KNeighborsRegressor(n_neighbors=5, weights='uniform') # chng
        algo_2 = Ridge(alpha=0.2) # chng
        algo_3 = Lasso(alpha=0.2, normalize=True) # chng   
        classifiers = [
            ('Linear Reg', algo_1),
            ('Ridge Reg', algo_2),
            ('Lasso Reg', algo_3)
            ]
    params = { # chng
        'kneighborsregressor__n_neighbors': range(2, 21),
        'kneighborsregressor__weights': ['uniform', 'distance']
        } 
    cv_number = 5
    scoring_method = 'neg_mean_squared_error'
    model = GridSearchCV(model, params, cv=cv_number, scoring=scoring_method)
    '''training'''
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        output_var,
        test_size=0.3,
        random_state=642,
        shuffle=True
        ) # chng

    '''tuning your model(s)'''

    # Voting Classifier
    if VC:
        for clf_name, clf in classifiers:
            clf.fit(X_train, y_train) 
            y_pred = clf.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Best Parameters chosen : {}".format(model.best_params_))
    
    '''evaluating your model(s)'''
    if R:
        '''regression metrics'''
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE: {:.2f}".format(rmse))

        r2 = r2_score(y_test,y_pred)
        print("R2 Score: {:.5f}".format(r2))
    
    else:
        '''classification metrics'''
        cm = confusion_matrix(y_test, y_pred)
        ax= plt.subplot()
        sn.heatmap(cm, annot=True, ax = ax, fmt="d")

        print(classification_report(y_test, y_pred, digits=2))
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    y_pred = model.predict(test_df)
    np.savetxt('C:/Users/nicho/Documents/GitHub/AnacondaML/SUP2/housing.csv', y_pred, delimiter=',')

Main()