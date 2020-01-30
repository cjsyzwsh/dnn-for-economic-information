from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys

# setup
run_dir = 'ldn_models/hyper_search'
numModels = 100
# London Dataset
variables = ['age', 'male', 'driving_license',
       'car_ownership', 'distance', 'dur_walking', 'dur_cycling',
       'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total',
       'pt_n_interchanges', 'dur_driving', 'cost_transit',
       'cost_driving_total']
#       'cost_driving_fuel', 'cost_driving_con_charge']

standard_vars = ['age', 'distance', 'dur_walking', 'dur_cycling',
       'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total',
       'pt_n_interchanges', 'dur_driving', 'cost_transit',
       'cost_driving_total']

standard_vars_idx = [0,4,5,6,7,8,9,10,11,12,13]
modes = ['walk', 'pt', 'cycle', 'drive']
data_dir = 'data/london/london_processed.pkl'
raw_data_dir =  'data/london/london_processed_raw.pkl'
suffix = ''
'''
# Singapore Dataset
variables = ['walk_walktime',
         'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt',
         'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt',
         'av_cost', 'av_waittime', 'av_ivt',
         'drive_cost', 'drive_walktime', 'drive_ivt',
         'age', 'inc', 'edu',
         'male','young_age', 'old_age',
         'low_edu', 'high_edu',
         'low_inc', 'high_inc',
         'full_job']
standard_vars = ['walk_walktime',
                 'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt',
                 'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt',
                 'av_cost', 'av_waittime', 'av_ivt',
                 'drive_cost', 'drive_walktime', 'drive_ivt',
                 'age', 'inc', 'edu']
standard_vars_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

modes = ["walk", "bus", "rideshare", "drive", "av"]
data_dir = 'data/processed_data/SGP.pickle'
raw_data_dir =  'data/processed_data/SGP_raw.pickle'
suffix = '_sp'
'''
m = Analysis(run_dir, numModels, variables, standard_vars_idx, modes, data_dir, suffix=suffix)
m.preprocess(raw_data_dir)
m.load_models_and_calculate(mkt=1, disagg=True, social_welfare=False,drive_time_idx=11, drive_cost_idx=13, drive_cost_idx_standard=10, \
                            drive_time_name='dur_driving', drive_cost_name='cost_driving_total', drive_idx=3, \
                            time_correction = 1)
# time_correction = 1 if time is in hours
# time_correction = 60 if time is in minutes
print(np.mean(m.mkt_share_train, axis = 0))
print(np.std(m.mkt_share_train, axis = 0))

print(np.mean(m.mkt_share_test, axis = 0))
print(np.std(m.mkt_share_test, axis = 0))

sys.exit()
print(np.mean(np.sum(m.sw_change_0, axis=1)))
print(np.mean(np.sum(m.sw_change_1, axis=1)))
print(np.mean(np.sum(m.sw_change_2, axis=1)))
print(np.mean(np.sum(m.sw_change_3, axis=1)))


with open(run_dir + '.pkl', 'wb') as f:
    pickle.dump(m, f)

drive_time = 13 - 3
drive_cost = 11 - 3

vot = m.util_derivative_test[drive_time,:,:] / m.util_derivative_test[drive_cost,:,:] / m.std[None, drive_time] * m.std[None, drive_cost] * time_correction
avg_vot = np.mean(m.vot_test, axis=1)
avg_vot = avg_vot[~np.isnan(avg_vot)]
print(np.mean(avg_vot))
print(np.median(avg_vot))
fig, ax = plt.subplots(figsize=(8, 8))
bins = np.linspace(np.percentile(avg_vot, 5), np.percentile(avg_vot, 95),
                   int((np.percentile(avg_vot, 95) - np.percentile(avg_vot, 5)) / 3))
ax.hist(avg_vot, bins=bins, density=True, color='b')
#ax.set_xlim([np.percentile(avg_vot, 5), np.percentile(avg_vot, 95)])
#ax.set_xlim([-200, 200])
ax.set_xlabel("Value of Time (Drive) ($/h)")
ax.set_ylabel('Density')

modelNumber = 3

filt = ~np.isnan(vot) & ~np.isinf(vot)
v = np.reshape(vot[filt], (numModels, -1))
print("Model ", modelNumber, " VOT (test): ", np.mean(v[modelNumber, :]))
print("Model ", modelNumber, " VOT (test median): ", np.median(v[modelNumber, :]))

np.min(vot[modelNumber, :])
