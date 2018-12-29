from multiprocessing import Process,Manager
from FFT import FFT
from scipy.io import arff
import pandas as pd
import numpy as np
import pandas as pd
import time,csv


metrics=['accuracy','recall','precision','false_alarm']

metrics_dic={'accuracy':-2,'recall':-6,'precision':-7,'false_alarm':-4}

start_time = time.time()
df1 = pd.read_csv('../lammps/lammps_1.csv')
df2 = pd.read_csv('../lammps/lammps_2.csv')
df3 = pd.read_csv('../lammps/lammps_3.csv')
df4 = pd.read_csv('../lammps/lammps_4.csv')
df5 = pd.read_csv('../lammps/lammps_5.csv')
df6 = pd.read_csv('../lammps/lammps_6.csv')
df7 = pd.read_csv('../lammps/lammps_7.csv')
df8 = pd.read_csv('../lammps/lammps_8.csv')
df9 = pd.read_csv('../lammps/lammps_9.csv')
frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9]
df = pd.concat(frames)
df.drop('Name',axis=1, inplace=True)
df_1 = pd.read_csv('../lammps/lammps_10.csv')
df_1.drop('Name',axis=1, inplace=True)
df_train = df
df_test = df_1

def FFT1():
    dic={}
    dic1={}
    for i in metrics:
        fft = FFT(max_level=5)
        fft.criteria= 'recall'
        fft.target = df.columns.values[-1]       
        training_df = pd.DataFrame(df_train)
        testing_df = pd.DataFrame(df_test)
        fft.train, fft.test = training_df, testing_df
        fft.build_trees()
        t_id = fft.find_best_tree()    
        fft.eval_tree(t_id)                
        dic[i]=fft.performance_on_test[t_id][metrics_dic[i]]        
    print([dic])
    end_time = time.time()
    print('Execution Time',end_time - start_time)

FFT1()





