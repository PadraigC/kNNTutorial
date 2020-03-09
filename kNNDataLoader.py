import pandas as pd
import numpy as np
from sklearn import preprocessing

def data_loader():
    Name_arr = []
    X_dir = {}
    y_dir = {}

    # Load CC Default dataset
    CC_data = pd.read_csv('CC_default.csv')

    CC_data.pop('ID')
    y = CC_data.pop('Default').values
    X = CC_data.values
    scaler = preprocessing.StandardScaler().fit(X) #A scaler object
    X_scaled = scaler.transform(X)

    Name_arr.append('Credit')
    X_dir['Credit'] = X_scaled.copy()
    y_dir['Credit'] = y.copy()

    # Load HTRU dataset
    h_names = ['X1','X2','X3','X4','X5','X6','X7','X8','Class']
    HTRU_df = pd.read_csv('HTRU_2.csv',index_col = False,names = h_names)
    HTRU_df.shape
    
    y = HTRU_df.pop('Class').values
    X = HTRU_df.values
    
    scaler = preprocessing.StandardScaler().fit(X) #A scaler object
    X_scaled = scaler.transform(X)
    X_scaled.shape

    Name_arr.append('HTRU')
    X_dir['HTRU'] = X_scaled.copy()
    y_dir['HTRU'] = y.copy()
    
    # Load Shuttle dataset
    s_names = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','Class']
    shuttle_df = pd.read_csv('shuttle.csv',sep=' ',names = s_names)

    # Remove instances of category with 6 instances
    shuttle_df= shuttle_df[ shuttle_df['Class'] != 6 ]
    shuttle_df['Class'].value_counts()
    
    y = shuttle_df.pop('Class').values
    X = shuttle_df.values
    
    scaler = preprocessing.StandardScaler().fit(X) #A scaler object
    X_scaled = scaler.transform(X)

    Name_arr.append('Shuttle')
    X_dir['Shuttle'] = X_scaled.copy()
    y_dir['Shuttle'] = y.copy()
    
    #Load Letter dataset
    l_names = ['Letter','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12',
          'X13','X14','X15','X16']
    letter_df = pd.read_csv('letter-recognition.csv', names = l_names)

    y = letter_df.pop('Letter').values
    X = letter_df.values
    
    scaler = preprocessing.StandardScaler().fit(X) #A scaler object
    X_scaled = scaler.transform(X)

    Name_arr.append('Letter')
    X_dir['Letter'] = X_scaled.copy()
    y_dir['Letter'] = y.copy()

    return Name_arr, X_dir, y_dir