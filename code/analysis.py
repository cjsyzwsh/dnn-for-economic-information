# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:36:14 2019

@author: wangqi44
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class Analysis:
    def __init__(self, run_dir, numModels, all_vars, cont_vars, modes, data_dir, suffix=''):
        # indexes of X_train_standard
        self.cont_vars = cont_vars
        # standard deviation of X_train_standard
        self.std = None
        # all variables' raw values
        self.X_train_raw = None
        # all continuous variables' raw values
        self.X_train_standard = None
        self.X_test_standard = None
        self.all_variables = all_vars
        self.modes = modes
        self.run_dir = run_dir
        self.numModels = numModels
        # standardized input feeding into NN
        self.numAlt = len(self.modes)
        self.colors = ['salmon', 'wheat','darkseagreen','plum','dodgerblue']

        self.suffix = suffix
        with open(data_dir, 'rb') as f:
            SGP = pickle.load(f)
        self.input_data = {}
        self.input_data['X_train'] = SGP['X_train'+self.suffix]
        self.input_data['Y_train'] = SGP['Y_train'+self.suffix]
        self.input_data['X_test'] = SGP['X_test'+self.suffix]
        self.input_data['Y_test'] = SGP['Y_test'+self.suffix]

        self.filterd_train = []
        self.filterd_test = []

        numContVars = len(self.cont_vars)
        self.numIndTrn = len(self.input_data['Y_train'])
        self.numIndTest = len(self.input_data['Y_test'])
        # (variable, model #, mode, individual #)
        self.mkt_prob_derivative = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        # mkt: market average individual, the individual dimension has only one variable; the others fixed at market average
        # not mkt: actual individuals with all the variables
        self.prob_derivative = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        self.elasticity = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        self.prob_derivative_test = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTest))
        self.util_derivative_test = np.zeros((numContVars, self.numModels, self.numIndTest))
        self.elasticity_test = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTest))
        # (variable, mode, individual #) - average choice prob
        self.average_cp = np.zeros((numContVars, self.numAlt, self.numIndTrn))
        # (variable, model #, mode, individual #) - choice prob
        self.mkt_choice_prob = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        self.vot_train = np.zeros((self.numModels, self.numIndTrn))
        self.vot_test = np.zeros((self.numModels, self.numIndTest))
        self.mkt_share_train = np.zeros((self.numModels, self.numAlt))
        self.mkt_share_test = np.zeros((self.numModels, self.numAlt))

    def compute_prob_curve(self, x, input_x, var, prob, sess):
        input_x = np.array(input_x)
        x_avg = np.mean(input_x, axis=0)
        x_feed = np.repeat(x_avg, len(input_x)).reshape(len(input_x), np.size(input_x, axis=1), order='F')
        choice_prob = []
        prob_derivative = []

        for idx in var:
            x_feed[:, idx] = input_x[:, idx]
            temp = sess.run(prob, feed_dict={x: x_feed})
            choice_prob.append(np.array(temp).T)
            temp = []
            for j in range(self.numAlt):
                grad = tf.gradients(prob[:, j], x)
                temp1 = sess.run(grad, feed_dict={x: x_feed})
                temp1 = temp1[0][:, idx]
                temp.append(temp1)
            prob_derivative.append(np.array(temp))
            x_feed[:, idx] = x_avg[idx]

        return np.array(choice_prob), np.array(prob_derivative)

    def load_models_and_calculate(self, mkt, disagg, social_welfare, drive_time_idx, drive_cost_idx, drive_cost_idx_standard, drive_time_name, drive_cost_name, drive_idx, time_correction):
        self.filterd_train = []
        self.filterd_test = []
        sw_ind = []
        for index in range(self.numModels):

            tf.reset_default_graph()

            sess = tf.InteractiveSession()
            saver = tf.train.import_meta_graph(self.run_dir + "/model" + str(index) + ".ckpt.meta")
            saver.restore(sess, self.run_dir + '/model' + str(index) + ".ckpt")

            graph = tf.get_default_graph()

            x = graph.get_tensor_by_name("X:0")
            prob = graph.get_tensor_by_name("prob:0")
            prob_train = sess.run(prob, feed_dict={x: self.input_data['X_train']})
            prob_test = sess.run(prob, feed_dict={x: self.input_data['X_test']})

            predict = tf.argmax(prob, axis=1)
            #predict_test = sess.run(predict, feed_dict={x:self.input_data['X_test']})
            utility = graph.get_tensor_by_name("logits:0")

            # market share
            self.mkt_share_train[index, :] = np.sum(prob_train, axis = 0) / self.numIndTrn
            self.mkt_share_test[index, :] = np.sum(prob_test, axis = 0) /self.numIndTest
            print(self.mkt_share_train[index, :])

            '''
            if disagg:
                # Disaggregate Prob Derivative and Elasticity
                grad = []
                drive_time = drive_time_idx
                drive_cost = drive_cost_idx
                for j in range(len(self.modes)):
                    grad.append(tf.gradients(prob[:, j], x))
                    gradients = sess.run(grad[-1], feed_dict={x: self.input_data['X_train']})[0]
                    # (variable, model #, mode, individual #) - elasticity/derivative
                    self.prob_derivative[:, index, j, :] = np.array(gradients[:, self.cont_vars]).T
                    elas = gradients[:, self.cont_vars] / prob_train[:, j][:, None] * np.array(self.X_train_standard)[:, self.cont_vars] / self.std[None,self.cont_vars]
                    self.elasticity[:, index, j, :] = np.array(elas).T

                    if j == drive_idx: # If drive, then calculate VOT
                        # filter out bad records
                        # print how many observations are filtered out
                        # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
                        # Take the mean of the trial
                        v = gradients[:, drive_time] / gradients[:, drive_cost] / self.std[None, drive_time] * self.std[None, drive_cost] * time_correction
                        self.vot_train[index,:] = v
                        filt = ~np.isnan(v) & ~np.isinf(v)
                        v = v[filt]
                        print("Model ", index, ": dropped ", self.numIndTrn - len(v), " training observations.")
                        self.filterd_train.append(self.numIndTrn - len(v))

                    gradients = sess.run(grad[-1], feed_dict={x: self.input_data['X_test']})[0]
                    self.prob_derivative_test[:, index, j, :] = np.array(gradients[:, self.cont_vars]).T
                    elas = gradients[:, self.cont_vars] / prob_test[:, j][:, None] * np.array(self.X_test_standard)[:, self.cont_vars] / self.std[None,self.cont_vars]
                    self.elasticity_test[:, index, j, :] = np.array(elas).T
                    
                    if j == 3: # If drive, then calculate VOT
                        # filter out bad records
                        # print how many observations are filtered out
                        # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
                        # Take the mean of the trial
                        v = gradients[:, drive_time] / gradients[:, drive_cost] / self.std[None, drive_time] * self.std[None, drive_cost] * time_correction
                        self.vot_test[index,:] = v
                        filt = ~np.isnan(v) & ~np.isinf(v)
                        v = v[filt]
                        print("Model ", index, ": dropped ", self.numIndTest - len(v), " testing observations.")
                        self.filterd_test.append(self.numIndTest - len(v))

                        util_derivative = tf.gradients(utility[:, j], x)
                        util_derivative_test = sess.run(util_derivative, feed_dict={x: self.input_data['X_test']})[0][:, self.cont_vars]
                        self.util_derivative_test[:, index, :] = np.array(util_derivative_test).T

            if mkt:
                # Choice prob and Prob Derivative for market average person
                choice_prob, prob_derivative = self.compute_prob_curve(x, self.input_data['X_train'], self.cont_vars, prob, sess)
                # (variable, model #, mode, individual #) - choice prob
                self.mkt_prob_derivative[:, index, :, :] = np.array(prob_derivative)
                self.mkt_choice_prob[:, index, :, :] = np.array(choice_prob)

            if social_welfare:
                utility0 = sess.run(utility, feed_dict={x: self.input_data['X_test']})
                new_input = self.input_data['X_test'].copy()
                new_input[drive_cost_name] += 1
                utility1 = sess.run(utility, feed_dict={x: new_input})
                sw_ind.append(np.log(np.sum(np.exp(utility1), axis=1)) - np.log(np.sum(np.exp(utility0), axis=1)))
            '''
            '''
            fig, ax = plt.subplots(figsize = (12, 12))
            ax.scatter(self.X_train_raw[:, 11], self.mkt_choice_prob[11, index, 3,:])
            fig.savefig('plots/test.png')
            '''

            sess.close()

        #self.average_cp = np.mean(self.mkt_choice_prob, axis = 1)
        if social_welfare:
            j = drive_idx
            cost_var = drive_cost_idx_standard
            self.sw_ind = np.array(sw_ind)
            self.sw_change_0 = sw_ind/self.util_derivative_test[cost_var,:,:] #, axis = 1)
            self.sw_change_1 = sw_ind/np.mean(self.util_derivative_test[cost_var,:,:], axis=0)[None, :]#, axis = 1)
            self.sw_change_2 = sw_ind/np.mean(self.util_derivative_test[cost_var,:,:], axis=1)[:, None]#, axis = 1)
            self.sw_change_3 = sw_ind/np.mean(self.util_derivative_test[cost_var,:,:])#, axis = 1)


    def preprocess(self, raw_data_dir, num_training_samples = None):
        with open(raw_data_dir, 'rb') as f:
            data = pickle.load(f)
        if num_training_samples is None:
            X_train_raw = data['X_train'+self.suffix].values
            X_test_raw = data['X_test'+self.suffix].values
        else:
            X_train_raw = data['X_train'+self.suffix].values[:num_training_samples, :]
            X_test_raw = data['X_test'+self.suffix].values[:num_training_samples]

        X_train_raw = pd.DataFrame(X_train_raw, columns = self.all_variables)
        X_test_raw = pd.DataFrame(X_test_raw, columns = self.all_variables)

        self.X_train_standard = X_train_raw
        self.X_test_standard = X_test_raw
        #X_train_nonstandard = X_train_raw[non_standard_vars]

        self.std = np.sqrt(StandardScaler().fit(pd.concat([self.X_train_standard, self.X_test_standard])).var_)
        #StandardScaler().fit(X_sp_train_standard).mean_[[8,9,10,14,15]]

        self.X_train_raw = np.array(X_train_raw)
