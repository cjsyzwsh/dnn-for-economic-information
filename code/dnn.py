#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:28:41 2019

@author: shenhao
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy

class FeedForward_DNN:
    def __init__(self,K,MODEL_NAME,INCLUDE_VAL_SET,INCLUDE_RAW_SET, RUN_DIR):
        self.graph = tf.Graph()
        self.K = K
        self.MODEL_NAME = MODEL_NAME
        self.INCLUDE_VAL_SET = INCLUDE_VAL_SET
        self.INCLUDE_RAW_SET=INCLUDE_RAW_SET
        self.RUN_DIR = RUN_DIR
#    MODEL_NAME = 'model'
#    INCLUDE_VAL_SET = False
#    input_file="data/SGP_SP.pickle"
#    INCLUDE_RAW_SET = True
#    self.K = 2
    
    def init_hyperparameter(self, rand):
        # h stands for hyperparameter
        self.h = {}
        if rand:
            self.h['M']=np.random.choice(self.hs['M_list'])
            self.h['n_hidden']=np.random.choice(self.hs['n_hidden_list'])
            self.h['l1_const']=np.random.choice(self.hs['l1_const_list'])
            self.h['l2_const']=np.random.choice(self.hs['l2_const_list'])
            self.h['dropout_rate']=np.random.choice(self.hs['dropout_rate_list'])
        else:
            self.h['M']=6
            self.h['n_hidden']=200
            self.h['l1_const']=1e-5#1e-10#1e-20
            self.h['l2_const']=0.001#0.001#1e-10
            self.h['dropout_rate']=0.01#1e-5
        self.h['batch_normalization']=True
        self.h['learning_rate']=1e-3
        self.h['n_iteration']=5000
        self.h['n_mini_batch']=200

    def change_hyperparameter(self, new_hyperparameter):
        assert bool(self.h) == True
        self.h = new_hyperparameter
    
    def init_hyperparameter_space(self):
        # hs stands for hyperparameter_space
        self.hs = {}
        self.hs['M_list'] = [1,2,3,4,5,6,7,8,9,10]
        self.hs['n_hidden_list'] = [25, 50, 100, 150, 200] # 6
        self.hs['l1_const_list'] = [0.1, 1e-2, 1e-3, 1e-5, 1e-10, 1e-20]# 8
        self.hs['l2_const_list'] = [0.1, 1e-2, 1e-3, 1e-5, 1e-10, 1e-20]# 8
        self.hs['dropout_rate_list'] = [1e-2, 1e-5] # 5
        self.hs['batch_normalization_list'] = [True, False] # 2
        self.hs['learning_rate_list'] = [0.01, 1e-3, 1e-4, 1e-5] # 5
        self.hs['n_iteration_list'] = [500, 1000, 5000, 10000] # 5
        self.hs['n_mini_batch_list'] = [50, 100, 200, 500] # 5        
    
    def random_sample_hyperparameter(self):
        assert bool(self.hs) == True
        assert bool(self.h) == True
        for name_ in self.h.keys():
            self.h[name_] = np.random.choice(self.hs[name_+'_list'])

    def obtain_mini_batch(self):
        index = np.random.choice(self.N_bootstrap_sample, size = self.h['n_mini_batch'])   
        self.X_batch = self.X_train_[index, :]
        self.Y_batch = self.Y_train_[index]
        
    def load_data(self, input_data, input_data_raw = None, num_training_samples = None):
        print("Loading datasets...")
        self.colnames = list(input_data['X_train'].columns)
        if num_training_samples is None:
            self.X_train = input_data['X_train'].values
            self.Y_train = input_data['Y_train'].values
        else:
            self.X_train = input_data['X_train'].values[:num_training_samples, :]
            self.Y_train = input_data['Y_train'].values[:num_training_samples]
        self.X_test=input_data['X_test'].values
        self.Y_test=input_data['Y_test'].values
        if self.INCLUDE_VAL_SET:
            self.X_val = input_data['X_val'].values
            self.Y_val = input_data['Y_val'].values

        if self.INCLUDE_RAW_SET:
            if num_training_samples is None:
                self.X_train_raw = input_data_raw['X_train'].values
                self.Y_train_raw = input_data_raw['Y_train'].values
            else:
                self.X_train_raw = input_data_raw['X_train'].values[:num_training_samples, :]
                self.Y_train_raw = input_data_raw['Y_train'].values[:num_training_samples]

            self.X_test_raw=input_data_raw['X_test'].values
            self.Y_test_raw=input_data_raw['Y_test'].values                
            if self.INCLUDE_VAL_SET:
                self.X_val_raw = input_data_raw['X_val'].values
                self.Y_val_raw = input_data_raw['Y_val'].values
                
        print("Training set", self.X_train.shape, self.Y_train.shape)
        print("Testing set", self.X_test.shape, self.Y_test.shape)
        if self.INCLUDE_VAL_SET:
            print("Validation set", self.X_val.shape, self.Y_val.shape)
        # save dim
        self.N_train,self.D = self.X_train.shape
        self.N_test,self.D = self.X_test.shape

    def bootstrap_data(self, N_bootstrap_sample):
        print("Bootstrap ", N_bootstrap_sample, " samples from training set...")
        self.N_bootstrap_sample = N_bootstrap_sample
        bootstrap_sample_index = np.random.choice(self.N_train, size = self.N_bootstrap_sample) 
        self.X_train_ = self.X_train[bootstrap_sample_index, :]
        self.Y_train_ = self.Y_train[bootstrap_sample_index]

    def standard_hidden_layer(self, name):
        # standard layer, repeated in the following for loop.
        self.hidden = tf.layers.dense(self.hidden, self.h['n_hidden'], activation = tf.nn.relu, name = name)
        if self.h['batch_normalization'] == True:
            self.hidden = tf.layers.batch_normalization(inputs = self.hidden, axis = 1)
        self.hidden = tf.layers.dropout(inputs = self.hidden, rate = self.h['dropout_rate'])

    def build_model(self):
        with self.graph.as_default():
            self.X = tf.placeholder(dtype = tf.float32, shape = (None, self.D), name = 'X')
            self.Y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y')
            self.hidden = self.X
            
            for i in range(self.h['M']):
                name = 'hidden'+str(i)
                self.standard_hidden_layer(name)
            # last layer: utility in choice models
            self.output=tf.layers.dense(self.hidden, self.K, name = 'output')
            self.prob=tf.nn.softmax(self.output, name = 'prob')
            self.output_tensor = tf.identity(self.output, name='logits')

            l1_l2_regularization = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.h['l1_const'], scale_l2=self.h['l2_const'], scope=None)
            vars_ = tf.trainable_variables()
            # weights = [var_ for var_ in vars_ if 'kernel' in var_.name]
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_l2_regularization, vars_)
            
            # loss function
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output, labels = self.Y), name = 'cost')
            self.cost += regularization_penalty # add l1 and l2 loss
            # evaluate
            correct = tf.nn.in_top_k(self.output, self.Y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            
            # 
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.h['learning_rate']) # opt objective
            self.training_op = self.optimizer.minimize(self.cost) # minimize the opt objective
            self.init = tf.global_variables_initializer()  
            self.saver= tf.train.Saver()
                        
    def train_model(self):
        with tf.Session(graph=self.graph) as sess:
            self.init.run()
            for i in range(self.h['n_iteration']):
                if i%500==0:
                    print("Iteration ", i, "Cost = ", self.cost.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_}))
                # gradient descent
                self.obtain_mini_batch()
                sess.run(self.training_op, feed_dict = {self.X: self.X_batch, self.Y: self.Y_batch})
            
            ## compute accuracy and loss
            self.accuracy_train = self.accuracy.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_})
            self.accuracy_test = sess.run(self.accuracy, feed_dict = {self.X: self.X_test, self.Y: self.Y_test})
            self.loss_train = self.cost.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_})
            self.loss_test = self.cost.eval(feed_dict = {self.X: self.X_test, self.Y: self.Y_test})
            if self.INCLUDE_VAL_SET:
                self.accuracy_val = self.accuracy.eval(feed_dict = {self.X: self.X_val, self.Y: self.Y_val})
                self.loss_val = self.cost.eval(feed_dict = {self.X: self.X_val, self.Y: self.Y_val})

            ## compute util and prob
            self.util_train=self.output.eval(feed_dict={self.X: self.X_train, self.Y: self.Y_train})
            self.util_test=self.output.eval(feed_dict={self.X: self.X_test, self.Y: self.Y_test})
            self.prob_train=self.prob.eval(feed_dict={self.X: self.X_train, self.Y: self.Y_train})
            self.prob_test=self.prob.eval(feed_dict={self.X: self.X_test, self.Y: self.Y_test})
            if self.INCLUDE_VAL_SET:
                self.util_val=self.output.eval(feed_dict={self.X: self.X_val, self.Y: self.Y_val})
                self.prob_val=self.prob.eval(feed_dict={self.X: self.X_val, self.Y: self.Y_val})
            ## save
            self.saver.save(sess, self.RUN_DIR+self.MODEL_NAME+".ckpt")

    def init_simul_data(self):
        self.simul_data_dic = {}

    def create_one_simul_data(self, x_col_name, x_delta):
        # create a dataset in which only targetting x is ranging from min to max. All others are at mean value.
        # add it to the self.simul_data_dic
        # use min and max values in testing set to create the value range
        target_x_index = self.colnames.index(x_col_name)
        self.N_steps = np.int((np.max(self.X_train[:,target_x_index]) - np.min(self.X_train[:,target_x_index]))/x_delta) + 1
        data_x_target_varying = np.tile(np.mean(self.X_train, axis = 0), (self.N_steps, 1))
        data_x_target_varying[:, target_x_index] = np.arange(np.min(self.X_train[:,target_x_index]), np.max(self.X_train[:,target_x_index]), x_delta)
        self.simul_data_dic[x_col_name] = data_x_target_varying
        
    def compute_simul_data(self):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, "tmp/"+self.MODEL_NAME+".ckpt")
            # compute util and prob
            self.util_simul_dic={}
            self.prob_simul_dic={}
            for name_ in self.simul_data_dic.keys():
                self.util_simul_dic[name_]=self.output.eval(feed_dict={self.X:self.simul_data_dic[name_]})
                self.prob_simul_dic[name_]=self.prob.eval(feed_dict={self.X:self.simul_data_dic[name_]})

    def init_x_delta_data(self):
        self.x_delta_data_dic = {}
        
    def create_one_x_delta_data(self, x_col_name, x_delta):
        # create a dataset in which only targetting x_col becomes x + delta. All the others are the SAME as x. 
        # by default, we focus on training set.
        # add the new dataset to the self.x_delta_data_dic
        target_x_index = self.colnames.index(x_col_name)
        x_delta_data = copy.copy(self.X_train)
        x_delta_data[:, target_x_index] = x_delta_data[:, target_x_index] + x_delta # add delta to the X_train dataset in x_col_name column
        self.x_delta_data_dic[x_col_name]=x_delta_data
        
    def compute_x_delta_data(self):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, "tmp/"+self.MODEL_NAME+".ckpt")
            # compute util and prob
            self.util_x_delta_dic={}
            self.prob_x_delta_dic={}
            for name_ in self.x_delta_data_dic.keys():
                self.util_x_delta_dic[name_]=self.output.eval(feed_dict={self.X:self.x_delta_data_dic[name_]})
                self.prob_x_delta_dic[name_]=self.prob.eval(feed_dict={self.X:self.x_delta_data_dic[name_]})
                            
#        self.INCLUDE_SIMUL_SET = True
    def visualize_choice_prob_function(self, x_col_name, color_list, label_list, xlabel, ylabel):
        assert len(color_list)==self.K
        assert len(label_list)==self.K
        # targeting x index.
        target_x_index = self.colnames.index(x_col_name)
#        color_list=['r','g','c','b','y']
#        label_list=['Walking','Bus','Ridesharing','Driving','AV']
        # plot
        fig = plt.figure(figsize = (12, 12))
        ax = plt.axes()
        # plot probability curves
        for prob_index in range(self.K):
            ax.plot(self.prob_simul_dic[x_col_name][:,prob_index], color=color_list[prob_index],alpha = 1, linewidth = 3, label = label_list[prob_index])
        ax.legend(loc = 1, fontsize = 'xx-large')
        ax.set_xticks(np.linspace(0, self.N_steps, 8)) # cut all x range into 8  segments.
        ax.set_xticklabels(np.round_(np.linspace(np.min(self.X_test_raw[:, target_x_index]), np.max(self.X_test_raw[:, target_x_index]), 8), decimals = 2))
        ax.set(xlabel=xlabel,ylabel=ylabel)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        ax.title.set_fontsize(20)
#        plt.savefig("../output/graph_driving_prob_sparse_"+"top"+str(len(indices))+".png")
#        plt.close()







