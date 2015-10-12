# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 18:18:14 2014

@author: Snigdha Chandan Khilar
"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
import math
import csv

def Read_data():
    print 'Reading Data from CSV file'
    X = np.loadtxt( '/HackTheTalk/TRAIN_FINAL.csv', delimiter=',')
    data_train=shuffle(X)
    X_train = data_train[:,0:39]
    Y_train = data_train[:,39]
    #print(X_train[0])
    #print(Y_train)
    print 'Reading Done'
    print 'Training classifier (this may take some time!)'
    gbc = GBC(n_estimators=58,max_depth=7,verbose=1)
    gbc.fit(X_train,Y_train)
    print('training done')
    filename='/HackTheTalk/EMOTIONCLASSIFIER.joblib.pkl'
    joblib.dump(gbc,filename)
    print('Save')
    
     
def main():
    Read_data()
    
main()