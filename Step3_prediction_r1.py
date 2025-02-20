import streprom
from streprom.constant import log_dir, predictor_trainop_dir
import os
import matplotlib as mpl
mpl.use('Agg')
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES']="0,1"


'''
Prediction-Round1
'''
#1.Train
train_data='datasets/sequence_activity/spe6_all_normalizationexp_train_data_notoohigh_andGEOafterBacPP_80bp_train_norepeat.npy' 
predictor_trainop_dir=log_dir+'/predictor_trainop'+str(train_data.split('_')[0])+'/'
pred= streprom.Predictors.CNN() 
pred.load_dataset(log_dir=predictor_trainop_dir,train_data=train_data,singleflag=True,train_test_ratio=0.9)
pred.BuildModel(DIM=128,batch_size=256,kernel_size=8)
#print(' [*]  Predictor training start...')
#pred.Train(epoch=10000,earlystop=20,log_dir=predictor_trainop_dir,checkpoint_dir=predictor_trainop_dir,lr=1e-6)#1e-5 #earlystop:300->20

#2.Load model
pred.load(checkpoint_dir = predictor_trainop_dir)
#3.Rank and scoring of generated candidates before testing in cells
pred.promoter_scoring_out(log_dir,'round1_generated.txt','round1_100_filtered_candidates',100)
#4. Rank and scoring of optimized candidates by Genetic Algorithm before testing in cells
pred.promoter_scoring_out(log_dir,'SeqIter70_fasta_byGA.txt','GA_finalorder_out.txt',100)
