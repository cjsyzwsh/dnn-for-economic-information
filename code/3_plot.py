# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:11:58 2019

@author: wangqi44
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def plot_choice_prob(m, plot_vars, plot_vars_standard, plot_var_names, highlight=[], highlightlabel=[]):
    colors = ['red','darkorange','darkgreen','darkorchid','blue']
    axes_cp = []
    for i in range(len(plot_vars)):
        axes_cp.append(plt.subplots(figsize=(8, 8)))
        mode = plot_var_names[i].find('_')
        axes_cp[i][1].set_ylabel(plot_var_names[i][:mode] + " probability")
        axes_cp[i][1].set_xlabel(plot_var_names[i])

    for i, v, vs in zip([x for x in range(len(plot_vars))], plot_vars, plot_vars_standard):
        mode = plot_var_names[i][:plot_var_names[i].find('_')]
        j = m.modes.index(mode)
        for index in range(m.numModels):
            plot = sorted(zip(m.X_train_raw[:, v], m.mkt_choice_prob[vs, index, j, :]))
            if index not in highlight:
                axes_cp[i][1].plot([x[0] for x in plot], [x[1] for x in plot], linewidth=1, color='silver',label='')
        for index in highlight:
            plot = sorted(zip(m.X_train_raw[:, v], m.mkt_choice_prob[vs, index, j, :]))
            axes_cp[i][1].plot([x[0] for x in plot], [x[1] for x in plot], linewidth=1, color=colors[highlight.index(index)], label = highlightlabel[highlight.index(index)])
        plot = sorted(zip(m.X_train_raw[:, v], m.average_cp[vs, j, :]))
        axes_cp[i][1].plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color='k',label='Average')
        axes_cp[i][1].set_xlim([0, np.percentile(m.X_train_raw[:, v], 99)])
        axes_cp[i][1].set_ylim([0, 1])
        axes_cp[i][1].legend(fancybox=True,framealpha = 0.5)
        axes_cp[i][0].savefig('plots/' + m.run_dir + '_choice_prob_' + str(v) + '.png', bbox_inches="tight")


def plot_substitute_pattern(m, plot_vars, plot_vars_standard, plot_var_names):
    colors = ['red','darkorange','darkgreen','darkorchid','blue']
    for name, v, vs in zip(plot_var_names, plot_vars, plot_vars_standard):
        fig, ax = plt.subplots(figsize=(8, 8))
        for j in range(m.numAlt):
            df = np.insert(m.mkt_choice_prob[vs, :, j, :], 0, m.X_train_raw[:, v], axis=0)
            df = df[:, df[0, :].argsort()]
            for index in range(m.numModels):
                ax.plot(df[0, :], df[index + 1, :], linewidth=1, alpha=0.5, color=m.colors[j], label='')
            
        for j in range(m.numAlt):
            plot = sorted(zip(m.X_train_raw[:, v], np.mean(m.mkt_choice_prob[vs, :, j, :], axis = 0)))
            ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color=colors[j], label=m.modes[j])
        #ax.legend()
        ax.set_ylabel("choice probability")
        ax.set_xlabel(name)
        ax.set_xlim([0, np.percentile(m.X_train_raw[:, v], 99)])
        ax.set_ylim([0, 1])
        fig.savefig('plots/' + m.run_dir + '_substitute_' + str(v) + '.png', bbox_inches="tight")


def plot_prob_derivative(m, plot_vars, plot_vars_standard, plot_var_names, highlight = [], highlightlabel = []):
    colors = ['red','darkorange','darkgreen','darkorchid','blue']
    axes_pd = []
    for i in range(len(plot_vars)):
        axes_pd.append(plt.subplots(figsize=(8, 8)))
        mode = plot_var_names[i].find('_')
        axes_pd[i][1].set_ylabel(plot_var_names[i][:mode] + " probability derivative")
        axes_pd[i][1].set_xlabel(plot_var_names[i])

    for i, v, vs in zip([x for x in range(len(plot_vars))], plot_vars, plot_vars_standard):
        mode = plot_var_names[i][:plot_var_names[i].find('_')]
        j = m.modes.index(mode)
        for index in range(m.numModels):
            # (variable, model  # , mode, individual #)
            df = pd.DataFrame(np.array([m.X_train_raw[:, v], m.mkt_prob_derivative[vs, index, j, :]]).T,
                              columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
            df.sort_values(by='var', inplace=True)
            if index not in highlight:
                axes_pd[i][1].plot(df['var'], df['pd'], linewidth=2, color='silver', label='')
        for index in highlight:
            df = pd.DataFrame(np.array([m.X_train_raw[:, v], m.mkt_prob_derivative[vs, index, j, :]]).T,
                              columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
            df.sort_values(by='var', inplace=True)
            axes_pd[i][1].plot(df['var'], df['pd'], linewidth=2, color=colors[highlight.index(index)], label = highlightlabel[highlight.index(index)])

        # interval = (np.percentile(m.X_train_raw[:, v], 95) - m.X_train_raw[:, v].min()) / 50
        # temp = m.X_train_raw[:, v] // interval * interval
        average_pd = np.mean(m.mkt_prob_derivative, axis=1)
        df = pd.DataFrame(np.array([m.X_train_raw[:, v], average_pd[vs, j, :]]).T, columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
        df.sort_values(by='var', inplace=True)
        axes_pd[i][1].plot(df['var'], df['pd'], linewidth=3, color='k',label='Average')
        axes_pd[i][1].set_xlim([0, np.percentile(m.X_train_raw[:, v], 99)])
        #axes_pd[i][1].set_ylim([-15, 5])
        axes_pd[i][1].legend(fancybox=True,framealpha = 0.5)
        axes_pd[i][0].savefig('plots/' + m.run_dir + '_prob_derivative_' + str(plot_vars[i]) + '.png', bbox_inches="tight")
    '''
    pos_1 = []
    pos_1_cnt = []
    neg_1 = []
    neg_1_cnt = []
    for j in range(5):
        neg_1.append(np.argmax(np.sum(rank_el[:, j, :] == 0, axis=1)))
        pos_1.append(np.argmax(np.sum(rank_el[:, j, :] == len(m.cont_vars) - 1, axis=1)))
        neg_1_cnt.append(np.max(np.sum(rank_el[:, j, :] == 0, axis=1)))
        pos_1_cnt.append(np.max(np.sum(rank_el[:, j, :] == len(m.cont_vars) - 1, axis=1)))
        print('Mode: ', m.mode[j], '\n\t', m.all_variables[pos_1[-1]], '(', pos_1_cnt[-1] / m.numModels, ')', \
              m.all_variables[neg_1[-1]], '(', neg_1_cnt[-1] / m.numModels, ')')
    '''


def plot_onemodel_pd(m, modelNumber, plot_var, plot_vars_standard, plot_var_names):
    for name, v, vs in zip(plot_var_names, plot_var, plot_vars_standard):
        fig, ax = plt.subplots(figsize=(8, 8))
        mode = name[:name.find('_')]
        j = m.modes.index(mode)
        temp = np.concatenate(
            (m.prob_derivative[vs, modelNumber, j, :], m.prob_derivative_test[vs, modelNumber, j, :]))
        bins = np.linspace(np.percentile(temp, 1), np.percentile(temp, 99), 20)
        ax.hist(m.prob_derivative[vs, modelNumber, j, :], bins=bins, density=True, color='b')
        ax.set_xlabel(name + " probability derivative")
        ax.set_ylabel('Density')
        ax.set_xlim([np.percentile(temp, 1), np.percentile(temp, 99)])
        fig.savefig('plots/' + m.run_dir + '_' + str(modelNumber) + '_prob_derivative_train_' + str(v) + '.png', bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(m.prob_derivative_test[vs, modelNumber, j, :], bins=bins, density=True, color='b')
        ax.set_xlabel(name + " probability derivative")
        ax.set_ylabel('Density')
        ax.set_xlim([np.percentile(temp, 1), np.percentile(temp, 99)])
        fig.savefig('plots/' + m.run_dir + '_' + str(modelNumber) + '_prob_derivative_test_' + str(v) + '.png', bbox_inches="tight")


def plot_onemodel_vot(m, modelNumber, currency):
    temp = np.concatenate((m.vot_test[modelNumber, :], m.vot_train[modelNumber, :]))
    bins = np.linspace(np.percentile(temp, 5), np.percentile(temp, 95),
                       int((np.percentile(temp, 95) - np.percentile(temp, 5)) / 3))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(m.vot_test[modelNumber, :], bins=bins, density=True, color='b')
    #ax.set_xlim([np.percentile(temp, 5), np.percentile(temp, 95)])
    #ax.set_xlim([-200, 200])
    ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
    ax.set_ylabel('Density')
    fig.savefig('plots/' + m.run_dir + '_' + str(modelNumber) + '_vot_drive_test.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(m.vot_train[modelNumber, :], bins=bins, density=True, color='b')
    print('(Train) Max Value:', np.max(m.vot_train[modelNumber, :]))
    print('(Train) Min Value:', np.min(m.vot_train[modelNumber, :]))
    print('(Train) 5th percentile Value:', np.percentile(m.vot_train[modelNumber, :], 5))
    print('(Train) 95th percentile Value:', np.percentile(m.vot_train[modelNumber, :], 95))
    print("Model ", modelNumber, " VOT mean: ", np.mean(m.vot_train[modelNumber, :]))
    print("Model ", modelNumber, " VOT median: ", np.median(m.vot_train[modelNumber, :]))
    print('(Test) Max Value:', np.max(m.vot_test[modelNumber, :]))
    print('(Test) Min Value:', np.min(m.vot_test[modelNumber, :]))
    print('(Test) 5th percentile Value:', np.percentile(m.vot_test[modelNumber, :], 5))
    print('(Test) 95th percentile Value:', np.percentile(m.vot_test[modelNumber, :], 95))
    print("Model ", modelNumber, " VOT (test mean): ", np.mean(m.vot_test[modelNumber, :]))
    print("Model ", modelNumber, " VOT (test median): ", np.median(m.vot_test[modelNumber, :]))
    #ax.set_xlim([np.percentile(temp, 5), np.percentile(temp, 95)])
    #ax.set_xlim([-200, 200])
    ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
    ax.set_ylabel('Density')
    fig.savefig('plots/' + m.run_dir + '_' + str(modelNumber) + '_vot_drive_train.png', bbox_inches="tight")

def plot_vot(m, currency):
    avg_vot = np.mean(m.vot_test, axis=1)
    avg_vot = avg_vot[~np.isnan(avg_vot)]
    print('Dropped ', m.numModels - len(avg_vot), ' Models.')
    print('Mean VOT test:', np.mean(avg_vot))
    print('Median VOT test:', np.median(avg_vot))
    fig, ax = plt.subplots(figsize=(8, 8))
    bins = np.linspace(np.percentile(avg_vot, 5), np.percentile(avg_vot, 95),
                       int((np.percentile(avg_vot, 95) - np.percentile(avg_vot, 5)) / 3))
    ax.hist(avg_vot, bins=bins, density=True, color='b')
    ax.set_xlim([np.percentile(avg_vot, 5), np.percentile(avg_vot, 95)])
    #ax.set_xlim([-200, 200])
    ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
    ax.set_ylabel('Density')
    fig.savefig('plots/' + m.run_dir + '_vot_drive_test.png', bbox_inches="tight")

    avg_vot_train = np.mean(m.vot_train, axis=1)
    avg_vot_train = avg_vot_train[~np.isnan(avg_vot_train)]
    print('Dropped ', m.numModels - len(avg_vot_train), ' Models.')
    print('Mean VOT train:', np.mean(avg_vot_train))
    print('Median VOT train:', np.median(avg_vot_train))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(avg_vot_train, bins=bins, density=True, color='b')
    ax.set_xlim([np.percentile(avg_vot, 5), np.percentile(avg_vot, 95)])
    #ax.set_xlim([-200, 200])
    ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
    ax.set_ylabel('Density')
    fig.savefig('plots/' + m.run_dir + '_vot_drive_train.png', bbox_inches="tight")

#run_dir = 'ldn_models/model15'
run_dir = 'sgp_models/model21'
plt.rcParams.update({'font.size': 24})
#currency = "£"
currency = "$"
with open(run_dir + '.pkl', 'rb') as f:
    m = pickle.load(f)

'''
plot_variables = ['bus_cost ($)', 'bus_ivt (min)', 'rideshare_cost ($)', 'rideshare_ivt (min)', \
                  'av_cost ($)', 'av_ivt (min)', 'drive_cost ($)', 'drive_ivt (min)']
plot_vars = [1, 4, 5, 7, 8, 10, 11, 13]
'''
plot_variables = ['drive_cost ($)']
plot_vars_standard = [10] # index of standard vars
plot_vars = [13]

plot_vot(m, currency)
assert len(plot_variables) == len(plot_vars) == len(plot_vars_standard)
#plot_choice_prob(m, plot_vars, plot_vars_standard, plot_variables)
#plot_prob_derivative(m, plot_vars, plot_vars_standard, plot_variables)#, [86,17,31], ['C1: 0.602', 'C2: 0.595', 'C3: 0.586'])
#plot_substitute_pattern(m, plot_vars, plot_vars_standard, plot_variables)
#plot_onemodel_pd(m, 78, plot_vars, plot_vars_standard, plot_variables)
plot_onemodel_vot(m, 3, currency)

'''
# average elasticity of invididuals
for i in np.arange(len(m.cont_vars)):
    print(m.all_variables[m.cont_vars[i]], end=' & ')
    for j in range(4):
        print("%.3f(%.1f)" % (np.mean(m.elasticity_test[i, 78, j, :]), np.std(m.elasticity_test[i, 78, j, :])), end = ' & ')
        #print(np.mean(m.elasticity[i, 1, 3, :]), np.std(m.elasticity[i, 1, 3, :]))
    print("\n")
    
# elasticity of average individual
for i, j in zip([0, 1, 4, 5, 7, 8, 10, 11, 13], 
                [10, 711, 210, 28, 178, 11, 4, 1234, 506]):
    pb = m.mkt_prob_derivative[i,:,3,j]
    ch = m.mkt_choice_prob[i,:,3, j]
    x = m.X_train_raw[j, i]
    std = m.std[i]
    elas = pb * x / ch / std
    print(np.mean(elas), np.std(elas))\
'''
print(np.mean(m.mkt_share, axis = 0))
print(np.std(m.mkt_share, axis = 0))

'''
print(np.mean(m.sw_change_1))
print(np.mean(m.sw_change_2))
print(np.mean(m.sw_change_3))
'''