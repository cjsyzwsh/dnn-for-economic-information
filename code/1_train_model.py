#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:19:07 2019

1_model training & analysis

@author: shenhao
"""

from dnn import FeedForward_DNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = 'ldn'

if dataset == 'sgp':
    # Load Singapore dataset 
    with open('data/processed_data/SGP.pickle', 'rb') as f:
        SGP = pickle.load(f)
    with open('data/processed_data/SGP_raw.pickle', 'rb') as f:
        SGP_raw = pickle.load(f)
        
    # adjust the dataset to fit the format
    input_data = {}
    input_data_raw = {}
    input_data['X_train'] = SGP['X_train_sp'][:1000]
    input_data['Y_train'] = SGP['Y_train_sp'][:1000]
    input_data['X_test'] = SGP['X_test_sp']
    input_data['Y_test'] = SGP['Y_test_sp']
    input_data_raw['X_train'] = SGP_raw['X_train_sp'][:1000]
    input_data_raw['Y_train'] = SGP_raw['Y_train_sp'][:1000]
    input_data_raw['X_test'] = SGP_raw['X_test_sp']
    input_data_raw['Y_test'] = SGP_raw['Y_test_sp']
    
if dataset == 'ldn':
    # Load London Dataset
    with open('data/london/london_processed.pkl', 'rb') as f:
        input_data = pickle.load(f)
    with open('data/london/london_processed_raw.pkl', 'rb') as f:
        input_data_raw = pickle.load(f)
        
num_models = 100
run_name = 'model15_7000'
num_alt = 4
INCLUDE_VAL_SET = False
INCLUDE_RAW_SET = True
N_bootstrap_sample = len(input_data['X_train'])
DIR = dataset + '_models/' + run_name + '/'
num_training_samples = 7000
df = []
for i in range(num_models):
    MODEL_NAME = 'model' + str(i)
    F_DNN = FeedForward_DNN(num_alt,MODEL_NAME,INCLUDE_VAL_SET,INCLUDE_RAW_SET, DIR)
    F_DNN.load_data(input_data, input_data_raw, num_training_samples)
    F_DNN.init_hyperparameter_space() # could change the hyperparameter here by using F_DNN.change_hyperparameter(new)
    F_DNN.init_hyperparameter(rand=False) # could change the hyperparameter here by using F_DNN.change_hyperparameter(new)
    F_DNN.bootstrap_data(N_bootstrap_sample)
    F_DNN.build_model()
    F_DNN.train_model()
    print(F_DNN.accuracy_train)
    print(F_DNN.accuracy_test)
    #print(F_DNN.prob_train[:, :5])
    #print(F_DNN.prob_test[:, :5])

    df.append([F_DNN.h['M'], F_DNN.h['n_hidden'], F_DNN.h['l1_const'], F_DNN.h['l2_const'], F_DNN.h['dropout_rate'], F_DNN.accuracy_train, F_DNN.accuracy_test])
    outfile = open('temp_result.txt', 'a')
    outfile.write("%d, %d, %.2E, %.2E, %.2E, %.4f, %.4f \n" % \
                  (F_DNN.h['M'], F_DNN.h['n_hidden'], F_DNN.h['l1_const'], F_DNN.h['l2_const'], F_DNN.h['dropout_rate'], F_DNN.accuracy_train, F_DNN.accuracy_test))
    outfile.close()

df = pd.DataFrame(df, columns = ['#layers','#hiddenunits','l1','l2','dropout','accuracy_train','accuracy_test'])
df.to_csv(dataset + '_models/' + run_name + '.csv')

