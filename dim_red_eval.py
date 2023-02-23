import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import time
from numpy.random import rand
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
import copy

def token_classifier_dim_red(x,y, dim_red_model, x_val = None, y_val = None, classifier_model = LogisticRegression(random_state = 0), smote = True):
# Evaluate using performance of token classifier
    # use trained dim red model
  
    dim_red_x = dim_red_model.transform(x)

    if smote:
        # SMOTE oversampling
        sm = SMOTE()
        X, Y = sm.fit_resample(dim_red_x, y)
    
    # if no SMOTE, just make X and Y copies of data going into smote
    else:
        X = dim_red_x.copy()
        Y = y.copy()

    # fit log reg model with smote balanced data
    model = classifier_model.fit(X, Y)


    # pred just on real data
    y_pred = model.predict(dim_red_x)
    
    # evaluate fit - train data
    acc = metrics.accuracy_score(y, y_pred)
    prec = metrics.precision_score(y, y_pred)
    reca = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    auc = metrics.roc_auc_score(y, y_pred)

    # evaluate fit - val data
    dim_red_x_val = dim_red_model.transform(x_val) 
    y_val_pred = model.predict(dim_red_x_val)

    val_acc = metrics.accuracy_score(y_val, y_val_pred)
    val_prec = metrics.precision_score(y_val, y_val_pred)
    val_reca = metrics.recall_score(y_val, y_val_pred)
    val_f1 = metrics.f1_score(y_val, y_val_pred)
    val_auc = metrics.roc_auc_score(y_val, y_val_pred)
    	
    return {'accuracy':acc, 'precision':prec,'recall':reca,'f1':f1, 'rocauc':auc,
    		'val_accuracy': val_acc, 'val_precision':val_prec,'val_recall':val_reca,'val_f1':val_f1, 'val_rocauc':val_auc }

    #else:
    	# output data dict
    #	return {'accuracy':acc, 'precision':prec,'recall':reca,'f1':f1, 'rocauc':auc, 'time':time.time() - start_t}


def make_cv_interpolate(data, holdout_size):
    '''Take a df and return a bootstrapped df of the same shape, with data replaced by interpolated values at random'''
    # make random 'bootstrap' mask with NaN values - CV values are considered the missing values - will be the error
    zero_mask = rand(*data.shape) > holdout_size
    train_mask = pd.DataFrame(zero_mask).astype(int).replace(0, np.nan)

    # combine data and 'train data' mask - need to do as np array multiplication so NaNs remain
    bootstrapped_df = np.array(train_mask) * np.array(data)
    # interpolate, columnwise on the missing values (NaN)
    bootstrapped_df = pd.DataFrame(bootstrapped_df, columns = data.columns)
    # train simple imputation
    imputed_data = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(bootstrapped_df)

    return imputed_data, zero_mask


def CV_train_error_dim_reduction(data,  model, holdout_size = 0.3):
    '''Model has to use scikit-learn API of fit'''
    # Create 'bootstrapped' data (cross val error)  - randomly chosen datapoints interpolated
    train_data, zero_mask = make_cv_interpolate(data, holdout_size)
    
    # copied so original model not transformed
    model_copy = copy.copy(model)
    
    # train dim. red. on bootstrapped data - predict input data
    transformed_matrix = model_copy.fit_transform(train_data)
    inverse_transform_matrix = model_copy.inverse_transform(transformed_matrix)
    
    print('Completed training - determining errors')
    
    # mean-square error of same data as original
    train_msq = mean_squared_error(np.array(data)[zero_mask], inverse_transform_matrix[zero_mask])
    # mean-square error of masked (randomly bootstrapped) data
    cv_msq = mean_squared_error(np.array(data)[~zero_mask], inverse_transform_matrix[~zero_mask])

    return train_msq, cv_msq 
