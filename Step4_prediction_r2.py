import streprom
from streprom.constant import log_dir,predictor_trainop_dir
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pickle
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES']="0,1"


'''
Prediction-Round2
'''
#1.Train
train_data='round1_experiment.txt' 
predictor_trainop_dir2 = log_dir+'/predictor_trainop'+str(train_data.split('_')[0])+'_2/'
pred= streprom.Predictors.CNN() 
pred.load_dataset(log_dir=predictor_trainop_dir,train_data=train_data,singleflag=True,train_test_ratio=0.9)#spe3_crosssignal_train70.npy  #seq_minexp_train70.npy
pred.BuildModel(DIM=128,batch_size=256,kernel_size=8)
#pred.Train(epoch=10000,earlystop=20,log_dir=predictor_trainop_dir,checkpoint_dir=predictor_trainop_dir2,lr=1e-6)

#2.Load model
pred.load(checkpoint_dir = predictor_trainop_dir2)
print(' [*] Predictor(CNN) loaded') 

#3.Evaluate 1 billion candidate sequences
filenamelist = ['genseq_onebillion_divide_1','genseq_onebillion_divide_2']
def seq2oh2(Seqs,charmap={'A':0,'T':1,'G':2,'C':3},num=4): #
    Onehot = []
    Length = 80
    for i in range(len(Seqs)):
        line = np.zeros([Length,num],dtype = 'float')
        if len(Seqs[i])==Length and set(Seqs[i]).issubset({'A', 'C', 'G', 'T'}): #len(Seqs[i])<=Length and 
            for j in range(len(Seqs[i])):
                line[j,charmap[Seqs[i][j]]] = 1
        Onehot.append(line)
    return Onehot

def promoter_out_finalround(filenamelist,out_filename,Noutbatch,Nout): #pred batch取2056
    seq_exp_all=pd.DataFrame(columns=['promoter','predicted_strength'])
    count=0
    for filename in filenamelist: #2个文件，每个文件里有92160*count这么多个序列
        print('filename:'+str(filename))
        with open(filename,'rb')as f:
            while True:
                count+=1
                try:
                    seqs = pickle.load(f)
                    seqs = seq2oh2(seqs) #92160
                    for i in range(36): #2560*36=92160 #2560*5=12800 #
                        inputseq=seqs[2560*i:2560*i+2559]
                        y=pred.Predictor(seq=inputseq,datatype='oh')
                        seq_exp=pd.DataFrame(columns=['promoter','predicted_strength'])
                        seq_exp['promoter']=inputseq #seqs[:len(y)]
                        seq_exp['predicted_strength']=y
                        seq_exp=seq_exp.sort_values(by=['predicted_strength'],ascending=False)
                        seq_exp = seq_exp.reset_index()
                        seq_exp_all=pd.concat([seq_exp_all,seq_exp.head(Noutbatch)],axis=0)#每个batch取前100条
                    if count%10==0:
                        print('iter(9766):'+str(count))
                        with open('./seq_exp_all_97600_1.29', 'wb') as f2:
                            pickle.dump(seq_exp_all, f2)
                except EOFError:
                    continue #break 
    #Total data(unsorted):
    with open('./seq_exp_all', 'wb') as f3:
        pickle.dump(seq_exp_all, f3)
    with open('./seq_exp_all','rb') as f:
        seq_exp_all = pickle.load(f) 
    print('[*] seq_exp_all loaded')
    seq_exp_all=seq_exp_all.sort_values(by=['predicted_strength'],ascending=False)
    seq_exp_all = seq_exp_all.reset_index()
    
    #Total data(sorted): 
    with open('./seq_exp_all_sorted', 'wb') as f4:
        pickle.dump(seq_exp_all, f4)
    #Filtering
    filtered_out=open(os.path.join(log_dir, out_filename),'w')
    outcount=0
    print('writing out seqs...')
    for i in range(Nout):
        outcount+=1
        filtered_out.write('>'+str(outcount)+'\n')
        filtered_out.write(str(seq_exp_all['promoter'][i])+'\n')
    filtered_out.close()
    print('[Done] Promoters written out.')
    return

promoter_out_finalround(log_dir,filenamelist,'round2_1billion_100_candidates.txt',100,100)




