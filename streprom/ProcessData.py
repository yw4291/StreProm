import os
import gzip
import struct
import tensorflow as tf
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def GetCharMap(seq):
    invcharmap = []
    for s in seq:
        for c in s:
            if c not in invcharmap:
                invcharmap += c
    charmap = {}
    count = 0
    for c in invcharmap:
        charmap[c] = count
        count += 1
    return charmap,invcharmap

def seq2oh(Seqs,charmap,num=4):
    Onehot = []
    Length = len(Seqs[0])
    for i in range(len(Seqs)):
        line = np.zeros([Length,num],dtype = 'float')
        for j in range(Length):
            line[j,charmap[Seqs[i][j]]] = 1
        Onehot.append(line)
    Onehot = np.array(Onehot)
    return Onehot

def oh2seq(oh,invcharmap):
    Seqs = []
    for i in range(oh.shape[0]):
        seq = str()
        for j in range(oh.shape[1]):
            seq = seq + invcharmap[np.argmax(oh[i,j,:])]
        Seqs.append(seq)
    return Seqs

def saveseq(filename,seq):
    f = open(filename,'w')
    for i in range(len(seq)):
        f.write('>'+str(i)+'\n')
        f.write(seq[i]+'\n')
    f.close()
    return

# def load_seq_data(filename,labelflag=0):#,nbin=3
#     if labelflag:
#         seq = []
#         label=[]
#         with open(filename,'r') as f:
#             for l in f:
#                 if l[0] == '>' or l[0] == '#':
#                     continue
#                 #
#                 l=l.strip('\n').split('\t')
#                 each_seq=l[0]
#                 #each_label=l[1]
#                 #
#                 #seq.append(str.strip(l))
#                 seq.append(each_seq)
#                 #
#                 #label.append(each_label)
#         charmap,invcharmap = GetCharMap(seq)
#         oh = seq2oh(seq,charmap)
#         return oh,charmap,invcharmap#,label
#     else:
#         seq = []
#         with open(filename,'r') as f:
#             for l in f:
#                 if l[0] == '>' or l[0] == '#':
#                     continue
#                 seq.append(str.strip(l))
#         charmap,invcharmap = GetCharMap(seq)
#         oh = seq2oh(seq,charmap)
#         return oh,charmap,invcharmap,seq

def load_seq_data(filename,labelflag=0):
    if labelflag:
        seq = []
        label=[]
        with open(filename,'r') as f:
            for l in f:
                if l[0] == '>' or l[0] == '#' or len(l)<2:
                    continue
                #
                l=l.strip('\n').split('\t')
                each_seq=l[0]
                #each_label=l[1]
                #
                #seq.append(str.strip(l))
                seq.append(each_seq)
                #
                #label.append(each_label)
        charmap,invcharmap = GetCharMap(seq)
        oh = seq2oh(seq,charmap)
        return oh,charmap,invcharmap#,label
    else:
        seq = []
        with open(filename,'r') as f:
            for l in f:
                if l[0] == '>' or l[0] == '#' or len(l)<2:
                    continue
                if set(str.strip(l)).issubset({'A', 'C', 'G', 'T'}):
                    seq.append(str.strip(l))
        charmap,invcharmap = GetCharMap(seq)
        oh = seq2oh(seq,charmap)
        return oh,charmap,invcharmap

def load_fun_data(filename):
    seq = []
    label = []
    with open(filename,'r') as f:
        for l in f:
            l = str.split(l)
            seq.append(l[0]) #1
            #print('seq:')
            #print(l[0])
            #
            #label.append(float(l[1])) #2
            label.append(np.log2(float(l[1]))) #2
            #print('label:')
            #print(float(l[1]))
    label = np.array(label)
    return seq,label

def load_fun_data_exp3(filename,filter=False,flag=None,already_log=False,getdata=False):#tissuechoose=0
    seq = []
    label = []
    seq_exp_bin=np.load(filename)
    if flag==1:
        if already_log:
            for i in range(len(seq_exp_bin)):
                seq.append(seq_exp_bin[i][0]) #少一个[[seq],[K562,HeLa,HepG2]]
                label.append(float(seq_exp_bin[i][1]))
            label = np.array(label)
            pdf = PdfPages('./log_10.22/input_minexp_distribution.pdf') 
            plt.figure(figsize=(7,7)) 
            plt.hist(label,40,density=1,histtype='bar',facecolor='blue',alpha=0.5)
            pdf.savefig() 
            pdf.close()
        else:
            for i in range(len(seq_exp_bin)):
                expraw=float(seq_exp_bin[i][1])+1 
                if expraw >-1 and math.log(expraw+1,2)>=0:
                    exp=math.log(expraw+1,2)
                    seq.append(seq_exp_bin[i][0]) #少一个[[seq],[K562,HeLa,HepG2]]
                    label.append(exp)
            label = np.array(label)
    else:
        for i in range(len(seq_exp_bin)):
            exp_multi_list=np.log2(np.array(seq_exp_bin[i][1])+1)
            if filter:
                if np.sum(np.array(exp_multi_list>0.5,dtype='float'))==3: #higher than 0.5 in all 3 tissues
                    seq.append(seq_exp_bin[i][0]) #1
                    label.append(exp_multi_list) #enhancer的mpra没加log2:label.append(seq_exp_bin[i][1]+1)
            else:
                seq.append(seq_exp_bin[i][0]) #1
                label.append(exp_multi_list) #enhancer的mpra没加log2:label.append(seq_exp_bin[i][1]+1)
        label = np.array(label)
    if getdata:
        return seq,label,seq_exp_bin
    else:
        return seq,label

###########################
class PromoterData(object): 
    def __init__(self, name, nbin,data_dir='',batch_dict_name=None,shuffle=True): #n_use_label=None, n_use_sample=None, batch_dict_name=None,
        assert os.path.isdir(data_dir)
        self._data_dir = data_dir
        self._shuffle = shuffle
        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name
        self.nbin=nbin

        assert name in ['train', 'test', 'val'] #判断输入的name是否为3个状态中的一个
        self.setup(epoch_val=0, batch_size=256)

        self._load_files(name) #n_use_label, n_use_sample
        self._seq_id_train = 0
        self._seq_id_test = 0

    def next_batch_dict(self,labeluse=False):
        batch_data = self.next_batch(labeluse=labeluse)
        data_dict = {key: data for key, data in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _load_files(self, name):#n_use_label, n_use_sample
        if name == 'train':
            #165bp:
            #unsupervise_seq = 'BS_EC_PA_JY_train_noagumentation.txt'
            #unsupervise_seq = 'BS_EC_PA_JY_train_comparative_genome_paper.txt'#'BS_EC_PA_JY_train_alltrain.txt'#all_180_seq_noN_final_short_10000_train.txt'
            #168bp:
            unsupervise_seq = 'BS_EC_PA_JY_all_comparative_genome_paper.txt'
            #supervise_file = 'BS_EC_PA_exp_bin_3_train50_originnobin.npy'
            supervise_file = 'BS_EC_PA_exp_bin_3_train80_originnobin.npy'
            #'BS_EC_PA_exp_bin_3_train50_originnobin_160.npy'
            ##BS_EC_PA_exp_bin_3_train50_originnobin.npy #'BS_EC_PA_exp_bin_3_train50.npy'#'EC_BS_PA_exp_bin_3_01_train_alltrain.npy'#'EC_BS_PA_exp_bin_3_train.npy' 
            # image_name = 'train-images-idx3-ubyte.gz'#'EC_BS_PA_exp_onehot.npy'
            # label_name = 'train-labels-idx1-ubyte.gz'
        else:
            unsupervise_seq = 'BS_EC_PA_JY_test.txt'#'all_180_seq_noN_final_short_10000_test.txt'
            supervise_file = 'EC_BS_PA_exp_bin_3_test.npy'
            # image_name = 't10k-images-idx3-ubyte.gz'
            # label_name = 't10k-labels-idx1-ubyte.gz'

        unsupervise_seq_path = os.path.join(self._data_dir, unsupervise_seq)
        supervise_file_path = os.path.join(self._data_dir, supervise_file)

        if 'label' in self._batch_dict_name or 'label_train' in self._batch_dict_name:
            seq_exp=np.load(supervise_file_path,allow_pickle=True)
            charmap={'A':0,'G':1,'C':2,'T':3}
            onehot = []
            label = []
            exp_all=[]
            for i in range(len(seq_exp)):
                exp_all.append(seq_exp[i][1])
            exp_all=sum(exp_all,[])
            #把exp_all分为n个bin，画个图
            pdf = PdfPages(os.path.join(self._data_dir, 'expdistri_forbin.pdf')) 
            plt.figure(1,figsize=(7,7)) 
            n, binlist, patches=plt.hist(exp_all, bins=self.nbin, color='green',alpha=0.5, rwidth=0.85)
            pdf.savefig() 
            pdf.close()
            if self.nbin==5:
                binlist=[-20,-10,-5,0,5,10] #去掉了-15这个bin
            else:
                binlist=list(binlist)#3个bin就按照正常分即可
            print('Expression bins:')
            print(binlist)
            binlist[-1]=binlist[-1]+1
            exp_bin=[]
            for i in range(len(seq_exp)):
                BS_1=seq_exp[i][1][0]#np.log2(seq_exp['tx_norm_BS'][i]+10**(-6))
                EC_1=seq_exp[i][1][1]#np.log2(seq_exp['tx_norm_EC'][i]+10**(-6))
                PA_1=seq_exp[i][1][2]#np.log2(seq_exp['tx_norm_PA'][i]+10**(-6))
                index_BS=np.where(BS_1>=binlist)[0][-1]
                index_EC=np.where(EC_1>=binlist)[0][-1]
                index_PA=np.where(PA_1>=binlist)[0][-1]#[-1]
                exp_bin.append([index_BS,index_EC,index_PA])

            for i in range(len(seq_exp)):
                seq=seq_exp[i][0]
                exp=exp_bin[i]
                #exp=seq_exp[i][1]
                eachseq = np.zeros([len(seq),4],dtype = 'float')
                for j in range(len(seq)):
                    base=seq[j]
                    eachseq[j,charmap[base]] = 1
                onehot.append(eachseq)
                label.append(np.array(exp))
            self.seq_list = np.array(onehot)
            self.label_list = np.squeeze(label) #(10185,3)
            print(np.unique(self.label_list,axis=0))
            print(len(np.unique(self.label_list,axis=0)))
            print(self.label_list.shape)
            #
            np.random.seed(3)
            seq_index_A = np.arange(self.seq_list.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.seq_list.shape[0]*int(0.9*10)//10
            self.seq_list_train, self.label_list_train = self.seq_list[seq_index_A[n:],:,:], self.label_list[seq_index_A[n:],:]#--2.[d:,:]
            self.seq_list_test, self.label_list_test = self.seq_list[seq_index_A[:n],:,:], self.label_list[seq_index_A[:n],:]#
            self.datanum_train=self.seq_list_train.shape[0]
            self.datanum_test=self.seq_list_test.shape[0]
            #
            self._suffle_files(labeluse=True)
        else:
            onehot = []
            label=[]
            seq=[]
            with open(unsupervise_seq_path,'r') as f:
                for l in f:
                    each_seq=l.strip('\n')
                    seq.append(each_seq)
            charmap,invcharmap = GetCharMap(seq)
            onehot = seq2oh(seq,charmap)
            self.seq_list = np.array(onehot)
            self._suffle_files()

    def _suffle_files(self,train_test_flag='train',labeluse=False):
        if self._shuffle:
            if train_test_flag=='train':
                np.random.seed(3)
                idxs = np.arange(self.datanum_train)#self.rng.shuffle(idxs)
                np.random.shuffle(idxs)
                self.seq_list_train = self.seq_list_train[idxs]
                #if labeluse:
                self.label_list_train = self.label_list_train[idxs]
            else:
                np.random.seed(3)
                idxs = np.arange(self.datanum_test)#self.rng.shuffle(idxs)
                np.random.shuffle(idxs)
                self.seq_list_test = self.seq_list_test[idxs]
                #if labeluse:
                self.label_list_test = self.label_list_test[idxs]
            

    def size(self,labeluse=False):
        return self.seq_list.shape[0]

    def next_batch(self,labeluse=False):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        #train
        start_train = self._seq_id_train
        self._seq_id_train += self._batch_size
        end_train = self._seq_id_train
        batch_seq_train = self.seq_list_train[start_train:end_train]
        #if labeluse:
        batch_label_train = self.label_list_train[start_train:end_train]
        #最开始shuffle train
        self._suffle_files(train_test_flag='train',labeluse=labeluse)
        if self._seq_id_train + self._batch_size > self.datanum_train:
            self._epochs_completed += 1
            self._seq_id_train = 0
            self._suffle_files(train_test_flag='train',labeluse=labeluse) 
        #test
        start_test = self._seq_id_test
        self._seq_id_test += self._batch_size
        end_test = self._seq_id_test
        batch_seq_test = self.seq_list_test[start_test:end_test]
        #if labeluse:
        batch_label_test = self.label_list_test[start_test:end_test]
        #最开始shuffle test
        self._suffle_files(train_test_flag='test',labeluse=labeluse)
        if self._seq_id_test + self._batch_size > self.datanum_test:
            self._epochs_completed += 1
            self._seq_id_test = 0
            #self._suffle_files(train_test_flag='test',labeluse=labeluse) 
        ###
        #if labeluse:
        return [batch_seq_train, batch_label_train,batch_seq_test, batch_label_test]
        #else:
        #return [batch_seq]

    def setup(self, epoch_val, batch_size, **kwargs):
        self.reset_epochs_completed(epoch_val)
        self.set_batch_size(batch_size)
        #self.reset_state()
        #try:
            #self._suffle_files()
        #except AttributeError:
            #pass

    def reset_epoch(self):
        self._epochs_completed = 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def reset_epochs_completed(self, epoch_val):
        self._epochs_completed  = epoch_val






# def oh2seq(oh,invcharmap):
#     Seqs = []
#     for i in range(oh.shape[0]):
#         seq = str()
#         for j in range(oh.shape[1]):
#             seq = seq + invcharmap[np.argmax(oh[i,j,:])]
#         Seqs.append(seq)
#     return Seqs


#def pdf2CDF(pdf):
#    CDF = pdf.copy()
#    for i in range(1,pdf.shape[0]):
#        CDF[i] = CDF[i] + CDF[i-1]
#    return CDF
#
#def Roulette(CDF):
#    temp = random.random()
#    for i in range(CDF.shape[0]):
#        if temp < CDF[i]:
#            return i

# def label_data_(supervise_file_path):
#     seq_exp=np.load(supervise_file_path,allow_pickle=True)
#     charmap={'A':0,'G':1,'C':2,'T':3}
#     onehot = []
#     label = []
#     for i in range(len(seq_exp)):
#         seq=seq_exp[i][0]
#         exp=seq_exp[i][1]
#         eachseq = np.zeros([len(seq),4],dtype = 'float')
#         for j in range(len(seq)):
#             base=seq[j]
#             eachseq[j,charmap[base]] = 1
#             onehot.append(eachseq)
#             label.append(np.array(exp))
#         self.seq_list = np.array(onehot)
#         self.label_list = np.squeeze(label) #(10185,3)
#         self._suffle_files(labeluse=True)
