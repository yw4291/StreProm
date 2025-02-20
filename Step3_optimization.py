import streprom
from streprom.constant import GA_dir
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES']="0,1"


'''
Optimization
'''
#1.Hyperparameters
save_freq_GA=10
save_freq_GD=2
Nrange=50 
Nrangemin=2 
MaxIter=5000

#2.Load generative model, predictive model and optimize algorithm
gen = streprom.Generators.WGAN_GP()
pred= streprom.Predictors.CNN() 
GA = streprom.Optimizers.GeneticAthm(gen.Generator,pred.Predictor,z_dim=128,onespeflag=True)
Nrangemin_GA = GA.run(outdir=GA_dir,save_freq=save_freq_GA,MaxIter=MaxIter) 
print(' [*] GA Optimizing seq ...')

#3.Evaluate afer each iteration               
def evalu_boxplot_linplot(result_dir,method,Nrange=Nrange,save_freq=save_freq_GA):
    #1.boxplot
    exp_GA_total=[]
    flist=[str((i+1)*save_freq) for i in range(Nrange)] 
    for fn in flist:
        exp_GA_total.append(np.load(result_dir+'ExpIter'+fn+'.npy')) 
        #exp_GA_total.append(np.power(2,np.load(result_dir+'ExpIter'+fn+'.npy')))
    exp_GA_total=np.array(exp_GA_total).transpose()
    plt.figure(figsize=(30, 10))
    pdf = PdfPages(result_dir+'/'+method+'_eachepoch_boxplot.pdf')
    plt.boxplot(exp_GA_total,showfliers=False)
    plt.grid(linestyle='--')
    pdf.savefig() 
    pdf.close()
    #lineplot
    exp_GA_total=[]
    flist=[str((i+1)*save_freq) for i in range(Nrange)] 
    for fn in flist:
        exp_GA_total.append(np.load(result_dir+'ExpIter'+fn+'.npy')) 
    exp_GA_total=np.array(exp_GA_total).transpose() 
    #
    print('gua')
    print(exp_GA_total.shape)
    minexp=np.min(exp_GA_total,0)
    maxexp=np.max(exp_GA_total,0)
    min5=np.percentile(exp_GA_total, 5,axis=0)
    max95=np.percentile(exp_GA_total, 95,axis=0)
    medianexp=np.median(exp_GA_total,0)
    plt.figure(figsize=(20, 10))
    pdf = PdfPages(result_dir+'/'+method+'_eachepoch_line.pdf')
    plt.plot(range(Nrange),medianexp,linewidth=2.5,color='black')
    plt.fill_between(range(Nrange), min5, max95, facecolor='blue', alpha=0.5)
    plt.fill_between(range(Nrange), minexp, maxexp, facecolor='blue', alpha=0.3)
    pdf.savefig() 
    pdf.close()
    return

evalu_boxplot_linplot(result_dir=GA_dir,method='GA',Nrange=Nrangemin_GA,save_freq=save_freq_GA) 

