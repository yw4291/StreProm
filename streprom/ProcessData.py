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
            seq.append(l[0]) 
            label.append(np.log2(float(l[1]))) 
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