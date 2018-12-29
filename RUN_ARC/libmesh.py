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
df1 = pd.read_csv('libmesh/libmesh_1.csv')
df2 = pd.read_csv('libmesh/libmesh_2.csv')
df3 = pd.read_csv('libmesh/libmesh_3.csv')
df4 = pd.read_csv('libmesh/libmesh_4.csv')
df5 = pd.read_csv('libmesh/libmesh_5.csv')
df6 = pd.read_csv('libmesh/libmesh_6.csv')
df7 = pd.read_csv('libmesh/libmesh_7.csv')
df8 = pd.read_csv('libmesh/libmesh_8.csv')
df9 = pd.read_csv('libmesh/libmesh_9.csv')
frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9]
df = pd.concat(frames)

df.drop('Name',axis=1, inplace=True)
df_1 = pd.read_csv('libmesh/libmesh_10.csv')
df_1.drop('Name',axis=1, inplace=True)

df_train = df
df_test = df_1


def FFT1(fft, i , return_dict):
    fft.build_trees1()
    t_id = fft.find_best_tree(0, 4)
    fft.eval_tree(t_id)
    return_dict[i] = fft.performance_on_test[t_id][metrics_dic[i]]

def FFT2(fft, i , return_dict):
    fft.build_trees2()
    t_id = fft.find_best_tree(4, 8)
    fft.eval_tree(t_id)
    return_dict[i] = fft.performance_on_test[t_id][metrics_dic[i]]

def FFT3(fft, i , return_dict):
    fft.build_trees3()
    t_id = fft.find_best_tree(8, 12)
    fft.eval_tree(t_id)
    return_dict[i] = fft.performance_on_test[t_id][metrics_dic[i]]


def FFT4(fft, i , return_dict):
    fft.build_trees4()
    t_id = fft.find_best_tree(12, 16)
    fft.eval_tree(t_id)
    return_dict[i] = fft.performance_on_test[t_id][metrics_dic[i]]


if __name__ == '__main__':
    dic = {}    
    fft = FFT(max_level=5)
    fft.criteria = 'recall'
    fft.target = df.columns.values[-1]
    training_df = pd.DataFrame(df_train)
    testing_df = pd.DataFrame(df_test)
    fft.train, fft.test = training_df, testing_df
    for i in metrics:
        with Manager() as manager:
            return_dict1 = manager.dict()
            return_dict2 = manager.dict()
            return_dict3 = manager.dict()
            return_dict4 = manager.dict()
            p1 = Process(target=FFT1, args=(fft,i,return_dict1))
            p2 = Process(target=FFT2, args=(fft,i,return_dict2))
            p3 = Process(target=FFT3, args=(fft, i, return_dict3))
            p4 = Process(target=FFT4, args=(fft, i, return_dict4))
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p1.join()
            p2.join()
            p3.join()
            p4.join()
            dic[i] = max(return_dict1.values()[0],return_dict2.values()[0],return_dict3.values()[0],return_dict4.values()[0])
    print(dic)
    end_time = time.time()
    print('Execution Time', end_time - start_time)





