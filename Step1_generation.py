import streprom
from streprom.constant import log_dir, generative_model_dir,sample_dir,evalu_dir,rep_dir,GA_dir,GD_dir,predictor_trainop_dir,predictor_evaluop_dir
import time
import os
import matplotlib as mpl
mpl.use('Agg')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES']="0,1"


mkdir=[log_dir,generative_model_dir,sample_dir,evalu_dir,rep_dir,GA_dir,GD_dir,predictor_trainop_dir,predictor_evaluop_dir]
for d in mkdir:
    if os.path.exists(d) == False:
        os.makedirs(d)
'''
generation
'''
input_file='datasets/input_promoter_sequences/spe6_NAR_primary_TK24_80bp_andGEOAfterBaPP_80bp_norepeatseq.txt'
gen = streprom.Generators.WGAN_GP(inputfile=input_file,SEQ_LEN=80,log_dir=log_dir)
gen.BuildModel(BATCH_SIZE=10240) 
ITERS=35000
#1.Train
#print(' [*]  Generator training start...')
#gen.Train(ITERS=ITERS,sample_dir=sample_dir,checkpoint_dir=generative_model_dir) 

#2.Load model
gen.load(global_step=ITERS-1,checkpoint_dir = generative_model_dir, model_name='wgan_gp')
print('[*] Generative model(WGAN-GP) loaded')

#3.Round1: generate initial pool
gen.Generator(Ntime=10,z=None,seed=0,n=2)
print('Round1 generation of candidate sequences is finished') 

#4.Round2: 1 billion candidate sequences
T1 = time.time()
#generate 
for i in range(4883): #102400*9766
    gen.Generator(Ntime=10,z=None,seed=i+4883,n=2)
    if i%20==0:
        T2= time.time()
        #Time log
        print(' [*] '+str(i)+'of 9766 have generated(2O i time):'+str(T2-T1)+'s:')
        T1=T2
print('Round2 generation (1 billion) of candidate sequences is finished') 

