#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 08:16:38 2018

Aim to clean dataset
Prepare for RP and SP datasets. 
Segment them into training, validation, and testing sets.

@author: shenhao
"""
#cd "/Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/4_dnn_emp_tb/code/code_paper2_empirical"

import numpy as np
import pandas as pd
import copy
import pickle

#import matplotlib.pyplot as plt
print("Start to clean the AV Singapore dataset...")

df_ind = pd.read_csv("data/initial_data/Individual_AV_Survey.csv")
df_alt = pd.read_csv("data/initial_data/Stand_alone_AV_SP_survey.csv")

#print(df_ind.columns)
#print(df_alt.columns)

lower_ind_columns = [col.lower() for col in df_ind.columns]
lower_alt_columns = [col.lower() for col in df_alt.columns]

df_ind.columns = lower_ind_columns
df_alt.columns = lower_alt_columns

# merge the two dataframes
df = pd.merge(df_alt, df_ind, on = 'id', how = 'inner')

#choice_values = [1,2,3,4,5]
#choice_alts = ['Walk', 'Bus', 'RideSharing', 'AV', 'Drive']
#df['choice'] = df['choice'].replace(to_replace = choice_values, value = choice_alts)

# obs where AV is not available but AVs are chosen; remove them 
df = df.loc[~np.logical_and(df['av_av'] == 0, df.choice == 4), :] # 4: av

############ edit values of variables ############
# sex
df['sex'] = df['sex'].replace(to_replace = [4, 5], value = ['male', 'female'])
# odd, one person has sex = 6, not willing to say. This person is super dif from all other people. Drop it.
df = df.loc[df.sex != 6, :]

df['male'] = (df['sex'] == 'male') * 1

# age 
df['young_age'] = (df.age < 35) * 1
df['old_age'] = (df.age > 60) * 1

# edu
df['low_edu'] = (df.edu < 3) * 1
df['medium_edu'] = ((df.edu > 2) & (df.edu < 5)) * 1
df['high_edu'] = (df.edu > 4) * 1 
                         
# income
inc_raw_values = [-1, 1, 2, 3, 4, 5, 6, 7, 9, 12, 23]
inc_final_values = [-1, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 17.5, 13.5, 0, 20.0] # thousand dollars
df['income'] = df['income'].replace(to_replace = inc_raw_values, value = inc_final_values)

df['low_inc'] = (df.income < 3) * 1
df['medium_inc'] = ((df.income > 2) & (df.income < 8)) * 1
df['high_inc'] = (df.income > 7) * 1 

# job
df['full_job'] = df['job']
full_job_raw_values = [1, 2, 3, 4, 5, 6, 7, 8, -1]
full_job_final_values = [1, 1, 0, 0, 0, 0, 0, 0, 0]
df['full_job'] = df['full_job'].replace(to_replace = full_job_raw_values, value = full_job_final_values)

### 
useful_vars = ['choice',
               'av_walk', 'av_bus', 'av_car', 'av_av', 'av_drive', 
               'walktime',
               'buscost', 'buswalk', 'buswait', 'busivt',
               'carcost', 'carwait', 'carivt', 
               'avcost', 'avwait', 'avivt',
               'drivcost', 'drivwalk', 'drivivt',
               'male', 
               'age', 'young_age', 'old_age',
               'edu', 'low_edu', 'high_edu', # medium edu skipped
               'income', 'low_inc', 'high_inc', # medium inc skipped
               'full_job' #, 'id', 'seq'
               ]

useful_vars_new_names = ['choice',
                         'available_walk', 'available_bus', 'available_ridesharing', 'available_av', 'available_drive',
                         'walk_walktime',
                         'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt', 
                         'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt', 
                         'av_cost', 'av_waittime', 'av_ivt', 
                         'drive_cost', 'drive_walktime', 'drive_ivt', 
                         'male', 
                         'age', 'young_age', 'old_age',
                         'edu', 'low_edu', 'high_edu',
                         'inc', 'low_inc', 'high_inc', 
                         'full_job' #, 'id', 'seq'
                         ]

df = df[useful_vars]
rename_dic = dict(zip(useful_vars, useful_vars_new_names))
df = df.rename(columns = rename_dic)

########################################################
# split rp and sp for nonstandard df
df_nonstand = copy.copy(df)

av_vars = ['available_walk', 'available_bus', 'available_ridesharing', 'available_av', 'available_drive']
rp_index = np.array(df_nonstand.available_walk == 1) & np.array(df_nonstand.available_bus == 1) & np.array(df_nonstand.available_ridesharing == 1) & \
                   np.array(df_nonstand.available_av == 0) & np.array(df_nonstand.available_drive == 1) 
sp_index = np.array(df_nonstand.available_walk == 1) & np.array(df_nonstand.available_bus == 1) & np.array(df_nonstand.available_ridesharing == 1) & \
                   np.array(df_nonstand.available_av == 1) & np.array(df_nonstand.available_drive == 1) 
df_nonstand['rp'] = np.int_(rp_index)
df_nonstand['sp'] = np.int_(sp_index)
df_nonstand_rp_sp = df_nonstand.loc[np.array(rp_index | sp_index), :]
df_nonstand_rp_sp = df_nonstand_rp_sp.drop(av_vars, axis = 1)

# replace choice index
old_alt_index = [1,2,3,4,5]
new_alt_index = [0,1,2,4,3]
df_nonstand_rp_sp['choice'] = np.int_(df_nonstand_rp_sp.choice.replace(to_replace = old_alt_index, value = new_alt_index))

df_nonstand_rp = df_nonstand_rp_sp.loc[df_nonstand_rp_sp.rp == 1, :]
df_nonstand_sp = df_nonstand_rp_sp.loc[df_nonstand_rp_sp.sp == 1, :]

df_nonstand_rp = df_nonstand_rp.drop(['rp','sp'], axis = 1)
df_nonstand_sp = df_nonstand_sp.drop(['rp','sp'], axis = 1)

#df_nonstand_rp.to_csv('../data/processed_data/data_AV_Singapore_v1_rp_full_nonstand.csv', index = False)
#df_nonstand_sp.to_csv('../data/processed_data/data_AV_Singapore_v1_sp_full_nonstand.csv', index = False)

############################################################
# normalize values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

standard_vars = ['walk_walktime',
                 'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt', 
                 'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt', 
                 'av_cost', 'av_waittime', 'av_ivt', 
                 'drive_cost', 'drive_walktime', 'drive_ivt', 
                 'age', 'inc', 'edu']
non_standard_vars = ['choice',
                     'available_walk', 'available_bus', 'available_ridesharing', 'available_av', 'available_drive',
                     'male',
                     'young_age', 'old_age',
                     'low_edu', 'high_edu',
                     'low_inc', 'high_inc', 
                     'full_job']

df_stand_part = pd.DataFrame(StandardScaler().fit_transform(df[standard_vars]), columns = standard_vars, index = df.index)
df_stand = pd.concat([df[non_standard_vars], df_stand_part], axis = 1)

############################################################
# transform the index of alternatives
# now: [1,2,3,4,5] meaning [walk,bus,ridesharing,av,drive]
# new: [0,1,2,3,4] meaning [walk,bus,ridesharing,drive,av]
old_alt_index = [1,2,3,4,5]
new_alt_index = [0,1,2,4,3]
df_stand['choice'] = np.int_(df_stand.choice.replace(to_replace = old_alt_index, value = new_alt_index))

############################################################
# split the raw df to RP and SP
av_vars = ['available_walk', 'available_bus', 'available_ridesharing', 'available_av', 'available_drive']
rp_index = np.array(df_stand.available_walk == 1) & np.array(df_stand.available_bus == 1) & np.array(df_stand.available_ridesharing == 1) & \
                   np.array(df_stand.available_av == 0) & np.array(df_stand.available_drive == 1) 
sp_index = np.array(df_stand.available_walk == 1) & np.array(df_stand.available_bus == 1) & np.array(df_stand.available_ridesharing == 1) & \
                   np.array(df_stand.available_av == 1) & np.array(df_stand.available_drive == 1) 
df_stand['rp'] = np.int_(rp_index)
df_stand['sp'] = np.int_(sp_index)
df_stand_rp_sp = df_stand.loc[np.array(rp_index | sp_index), :]
df_stand_rp_sp = df_stand_rp_sp.drop(av_vars, axis = 1)

df_stand_rp = df_stand_rp_sp.loc[df_stand_rp_sp.rp == 1, :]
df_stand_sp = df_stand_rp_sp.loc[df_stand_rp_sp.sp == 1, :]

df_stand_rp = df_stand_rp.drop(['rp','sp'], axis = 1)
df_stand_sp = df_stand_sp.drop(['rp','sp'], axis = 1)

############################################################
# rearange names
names = ['choice','walk_walktime',
         'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt', 
         'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt', 
         'av_cost', 'av_waittime', 'av_ivt', 
         'drive_cost', 'drive_walktime', 'drive_ivt', 
         'age', 'inc', 'edu', 
         'male','young_age', 'old_age',
         'low_edu', 'high_edu',
         'low_inc', 'high_inc', 
         'full_job']

df_nonstand_rp=df_nonstand_rp[names]
df_nonstand_sp=df_nonstand_sp[names]
df_stand_rp=df_stand_rp[names]
df_stand_sp=df_stand_sp[names]

# rearrange index and split them.
df_nonstand_rp.index=np.arange(0,df_nonstand_rp.shape[0])
df_nonstand_sp.index=np.arange(0,df_nonstand_sp.shape[0])
df_stand_rp.index=np.arange(0,df_stand_rp.shape[0])
df_stand_sp.index=np.arange(0,df_stand_sp.shape[0])

rp_index = np.arange(df_stand_rp.shape[0])
np.random.shuffle(rp_index)
sp_index = np.arange(df_stand_sp.shape[0])
np.random.shuffle(sp_index)

training_rp_index, testing_rp_index = \
    np.split(rp_index,[np.int_(5/6*df_stand_rp.shape[0])])
training_sp_index, testing_sp_index = \
    np.split(sp_index,[np.int_(5/6*df_stand_sp.shape[0])])

############################################################
### divide the full datasets into
## 2*2*2*2
# X_train_rp
# Y_train_rp
# X_test_rp
# Y_test_rp
# X_train_sp
# Y_train_sp
# X_test_sp
# Y_test_sp

# X_train_rp_raw
# Y_train_rp_raw
# X_test_rp_raw
# Y_test_rp_raw
# X_train_sp_raw
# Y_train_sp_raw
# X_test_sp_raw
# Y_test_sp_raw

SGP = {}
SGP['X_train_rp']=df_stand_rp.iloc[training_rp_index,1:]
SGP['Y_train_rp']=df_stand_rp.iloc[training_rp_index,0]
SGP['X_test_rp']=df_stand_rp.iloc[testing_rp_index,1:]
SGP['Y_test_rp']=df_stand_rp.iloc[testing_rp_index,0]
SGP['X_train_sp']=df_stand_sp.iloc[training_sp_index,1:]
SGP['Y_train_sp']=df_stand_sp.iloc[training_sp_index,0]
SGP['X_test_sp']=df_stand_sp.iloc[testing_sp_index,1:]
SGP['Y_test_sp']=df_stand_sp.iloc[testing_sp_index,0]
SGP_raw = {}
SGP_raw['X_train_rp']=df_nonstand_rp.iloc[training_rp_index,1:]
SGP_raw['Y_train_rp']=df_nonstand_rp.iloc[training_rp_index,0]
SGP_raw['X_test_rp']=df_nonstand_rp.iloc[testing_rp_index,1:]
SGP_raw['Y_test_rp']=df_nonstand_rp.iloc[testing_rp_index,0]
SGP_raw['X_train_sp']=df_nonstand_sp.iloc[training_sp_index,1:]
SGP_raw['Y_train_sp']=df_nonstand_sp.iloc[training_sp_index,0]
SGP_raw['X_test_sp']=df_nonstand_sp.iloc[testing_sp_index,1:]
SGP_raw['Y_test_sp']=df_nonstand_sp.iloc[testing_sp_index,0]

# save 
with open('data/processed_data/SGP.pickle', 'wb') as f:
    pickle.dump(SGP, f)
with open('data/processed_data/SGP_raw.pickle', 'wb') as f:
    pickle.dump(SGP_raw, f)



