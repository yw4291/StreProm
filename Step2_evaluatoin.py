import streprom
import shutil
from streprom.constant import log_dir, sample_dir,evalu_dir,promoter_len
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

#1.Evaluation file sequence to fasta
input_file='spe6_NAR_primary_TK24_80bp_andGEOAfterBaPP_80bp_norepeatseq.txt' 
n_model_epoch=33599
evalu_file_input='samples_'+str(n_model_epoch)+'.txt'
f = open(sample_dir+'/'+evalu_file_input)
evalu_file='samples_'+str(n_model_epoch)+'_nounk'
fout = open(sample_dir+'/'+evalu_file+'.txt','w')
count=0
for line in f.readlines():
    seq = line.strip('\n')
    if '>' not in line and set(seq).issubset({'A','G','C','T'}):
        count+=1
        fout.write('>'+str(count)+'\n')
        fout.write(seq+'\n')
    
f.close()
fout.close()

#2.GC content, ployA/G/C/T distribution and kmer correlation plot
source = sample_dir+'/'+evalu_file+'.txt' 
target = log_dir 
shutil.copy(source, target) 
evalu_file='samples_'+str(n_model_epoch)+'_nounk'
color_list=['black','brown','darkblue','purple','grey','yellow','blue','red','orange','green']
species_list=[input_file.strip('.txt'),evalu_file] 
numk=6
colorList=['orange'] 
evalu=streprom.Evaluation.evalu(data_dir=log_dir,save_dir=evalu_dir)
evalu.GC_plot(species_list,color_list)   
evalu.polyN_plot(species_list,color_list,4,10)
evalu.kmer_statistic(species_list,numk=numk)
evalu.kmer_plot(species_list,input_file.strip('.txt'),'cor',colorList,species_list=species_list,numk=numk)  

#3.-10 and -35 region fasta
f=open(log_dir+'/samples_'+str(n_model_epoch)+'_nounk.txt','r')
fout1=open(evalu_dir+'samples_'+str(n_model_epoch)+'_fasta_10region_nounk.txt','w')
fout2=open(evalu_dir+'samples_'+str(n_model_epoch)+'_fasta_35region_nounk.txt','w')
count=0
for line in f.readlines():
    if '>' not in line:
        fout1.write('>'+str(count)+'\n')
        #-10: −20 to +1 (-10和-35位置参照 2022-Genome-scale analysis of genetic regulatory elements in Streptomyces avermitilis MA-4680 using transcript boundary information)
        fout1.write(line.strip('\n')[(promoter_len-50):]+'\n') # -20   #50->80=30
        fout2.write('>'+str(count)+'\n')
        #-35:−25 to -40
        fout2.write(line.strip('\n')[(promoter_len-41):(promoter_len-24)]+'\n') #[(promoter_len-41):(promoter_len-24)]    
        count+=1
f.close()
fout1.close()
fout2.close()  