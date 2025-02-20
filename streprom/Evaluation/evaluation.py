import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle
from matplotlib.backends.backend_pdf import PdfPages


#Evaluation of GC,polyA/G/C/T,K-mer
class evalu: 
    def __init__(self, 
                 data_dir='./HierDNA/hier_input/',
                 save_dir='./HierDNA/hier_output/'
                 ):
        self.data_dir=data_dir
        self.save_dir=save_dir

    def remove_nan_inf(self,data,col):
        for i in range(len(data)):
            if str(data[col][i])=='nan' or str(data[col][i])=='inf':
                data[col][i]=0
        return data
    
    #1.GC content
    def seq_GC(self,seq):
        GCcount=0
        for i in range(len(seq)): 
            if(seq[i])=='G'or (seq[i])=='C':
                GCcount+=1
        return GCcount/len(seq)*100

    def GC_content(self,filename):
        fasta_file = self.data_dir+'/'+filename 
        with open(fasta_file, 'r') as f:
            GC=[]
            for line in f.readlines():
                if not line.startswith(">"):
                    seq = line.strip()
                    GC.append(self.seq_GC(seq))
        return np.array(GC)

    def GC_plot(self,species_list,color_list):
        print(' [*] Plot GC criterion...')
        pdf = PdfPages(self.save_dir+'GC_content_'+str(species_list[1])+'_'+str(species_list[0])+'.pdf') 
        plt.figure(figsize=(7,7)) 
        GC_result = locals()
        for i in range(len(species_list)):
            spe=species_list[i]
            filename=spe+'.txt' 
            GC_result[i]=self.GC_content(filename)
            sns.distplot(GC_result[i],kde=True,norm_hist=True,color=color_list[i])
        plt.legend(labels=species_list,loc='upper left')
        plt.xlabel('GC content(%)')
        plt.ylabel('Frequency')
        pdf.savefig() 
        pdf.close()
        
    #2.polyA/G/C/T
    def polyN_seq(self,seq,min,max):
        polyN=np.zeros(shape=((max-min+1),1))
        kmern=np.zeros(shape=((max-min+1),1))
        baselist=['A','G','C','T']
        for n in range(min,max+1,1):
            kmern[n-min,]+=len(seq)-n+1
            for i in range(len(seq)-n+1):
                for base in baselist:
                    if seq[i:i+n]==str(base)*n:
                        polyN[n-min,]+=1
        return kmern,polyN

    def polyN_count(self,filename,min,max):
        fasta_file = self.data_dir+'/'+filename
        with open(fasta_file, 'r') as f:
            kmern_allseq=np.zeros(shape=((max-min+1),1))
            polyN_count=np.zeros(shape=((max-min+1),1))
            for line in f.readlines():
                if not line.startswith(">"):
                    seq = line.strip()
                    kmern,polyN_count_add=self.polyN_seq(seq,min,max)
                    polyN_count+=polyN_count_add
                    kmern_allseq+=kmern
        return np.array(polyN_count)/kmern_allseq

    def polyN_plot(self,species_list,color_list,min,max):
        print(' [*] Plot ployN criterion...')
        pdf = PdfPages(self.save_dir+'/PolyN_'+str(species_list[1])+'.pdf') 
        plt.figure(figsize=(7,7)) 
        data=pd.DataFrame([])
        for i in range(len(species_list)):
            spe=species_list[i]
            filename= spe+'.txt'
            PloyN_result=self.polyN_count(filename,min,max).reshape((-1))
            PloyN_result=pd.DataFrame([PloyN_result,np.array([str(spe)]*(max-min+1))]).T
            data=pd.concat([data,PloyN_result])
        data.columns=['number','Species']
        sns.barplot(x=np.array(data.index)+min,y='number',hue='Species',data=data,
                    alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel('PolyN')
        plt.ylabel('Frequency')
        pdf.savefig() 
        pdf.close()

    #3.kmer
    def count_kmer(self,seq,kmers,numk):
        k = numk
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N'not in kmer:
                if kmer in kmers:
                    kmers[kmer] += 1
                else:
                    kmers[kmer] = 1
        sortedKmer = sorted(kmers.items(), reverse=True)
        return sortedKmer,kmers
            
    def kmer_fasta(self,filename,numk):
        input_fasta_file=self.data_dir+'/'+filename
        kmer_result= pd.DataFrame([])
        count=[]
        kmerseq=[]
        with open(input_fasta_file, 'r') as f:
            seq = ""
            kmers={}
            for line in f.readlines():
                if not line.startswith(">"):
                    seq = line.strip()
                    sortedKmer,kmer=self.count_kmer(seq,kmers,numk)
        for key,value in sortedKmer:
            kmerseq.append(key)
            count.append(value)
        kmer_result['kmer']=kmerseq
        kmer_result['count']=count
        kmer_sort_plot=kmer_result.sort_values(by=['count'])
        return kmer_sort_plot

    def kmer_statistic(self,species_list,numk=6):
        data=pd.DataFrame([])
        for i in range(len(species_list)):
            spe=species_list[i]
            filename= spe+'.txt'
            kmer_sort_plot=self.kmer_fasta(filename,numk)
            kmer_sort_plot.columns=['kmer',str(spe)]
            if data.shape!=(0, 0):
                data=pd.merge(data,kmer_sort_plot,how='left',on='kmer')
            else:
                data=kmer_sort_plot
            kmer_compare=self.remove_nan_inf(data,str(spe))    
        f = open(self.save_dir+'/kmer_result_'+str(species_list[1])+'_'+str(numk)+'.pkl', 'wb')
        pickle.dump(kmer_compare, f)
        print(' [*] Evalu kmer cal done')
        f.close()
            
    def kmer_plot(self,draw_list,sort_by_species,plt_type,color_list,species_list,numk=6):
        print(' [*] kmer correlation criterion...')
        kmer_compare = pickle.load(open(self.save_dir+'/kmer_result_'+str(species_list[1])+'_'+str(numk)+'.pkl', 'rb'))
        kmer_compare=kmer_compare.sort_values(by=[str(sort_by_species)])
        kmer_compare=kmer_compare.reset_index(drop=True)
        print(kmer_compare)
        y = locals()
        for i in range(len(draw_list)):
            y[i]=np.array(kmer_compare[draw_list[i]])/np.sum(np.array(kmer_compare[draw_list[i]]))*100
        if plt_type =='cor':
            pdf = PdfPages(self.save_dir+'/kmer_cor_'+str(numk)+'_'+str(draw_list[0])+'_'+str(draw_list[1])+'.pdf') 
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.scatter(y[0],y[1],color=str(color_list[0]),s=7,alpha=0.5,marker='.')
            plt.xlim(min(min(y[0]),min(y[1]))-0.02,max(max(y[0]),max(y[1])))
            plt.ylim(min(min(y[0]),min(y[1]))-0.02,max(max(y[0]),max(y[1])))
            r=pearsonr(y[0],y[1])[0]
            print(r)
            r=("%.3f" % r )
            plt.text(max(y[0])*0.8,max(y[1])*0.8,r'r='+str(r))
            plt.xlabel(str(draw_list[0]))
            plt.ylabel(str(draw_list[1]))
            pdf.savefig() 
            pdf.close()      
        elif plt_type =='freq':
            pdf = PdfPages(self.save_dir+'result_pdf/'+str(sort_by_species)+'_sort_'+str(draw_list[1])+'_'+str(numk)+'mer.pdf')
            fig, ax = plt.subplots(figsize=(5, 5))
            x=np.array(range(len(kmer_compare)))
            for i in range(len(draw_list)):
                plt.scatter(x,y[i],color=str(color_list[i]),s=7,alpha=0.5,marker='.',label=str(draw_list[i]))
            plt.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,ncol=1, markerfirst=True, markerscale=4, numpoints=1, handlelength=3.5)
            plt.xlabel(str(numk)+'-mers')
            plt.ylabel('Frequency(%)')
            pdf.savefig() 
            pdf.close() 
        return r
