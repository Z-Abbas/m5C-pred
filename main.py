#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 08:58:17 2023

@author: zeeshan
"""

import argparse
import pandas as pd
import numpy as np
import re, os, sys
import shap
import pickle
import random
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, roc_auc_score, auc
from sklearn.metrics import matthews_corrcoef

# Add the remaining code here
def read_nucleotide_sequences(file):
    
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]

    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences



def CKSNAP(fastas, gap, **kw):
  
    kw = {'order': 'ACGT'}
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#', 'label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings


def NCP(fastas, **kw):

    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) * 3 + 1):
        header.append('NCP.F'+str(i))
    encodings.append(header)
    chemical_property = {
        'A': [1, 1, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0],
        'T': [0, 0, 1],
        'U': [0, 0, 1],
        '-': [0, 0, 0],
    }
    names = []
    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for aa in sequence:
            code = code + chemical_property.get(aa, [0, 0, 0])
        encodings.append(code)
        names.append(name)
    return encodings, names

from Bio import SeqIO
def dataProcessing(path,fileformat):
    all_seq_data = []
    all_seq_data3 = []

    for record in SeqIO.parse(path,fileformat):
        sequences = record.seq # All sequences in dataset
    
        seq_data=[]
        seq_data3=[]
         
        
        for i in range(len(sequences)):
            if sequences[i] == 'A':
                seq_data3.append([1])
            if sequences[i] == 'T':
                seq_data3.append([2])
            if sequences[i] == 'U':
                seq_data3.append([2])
            if sequences[i] == 'C':
                seq_data3.append([3])
            if sequences[i] == 'G':
                seq_data3.append([4])
            if sequences[i] == '0':
                seq_data3.append([0])        
        all_seq_data3.append(seq_data3)
        
    all_seq_data = np.array(all_seq_data3);
    
    return all_seq_data
    

#------------------------------

from collections import Counter

def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

def PseEIIP(fastas, **kw):
    for i in fastas:
        if re.search('[^ACGTUacgtu-]', i[1]):
            print('Error: illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.')
            return 0

    base = 'ACGT'

    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335,
        'U': 0.1335
    }

    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    EIIPxyz = {}
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

    encodings = []
    header = ['#', 'label'] + trincleotides
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        trincleotide_frequency = TriNcleotideComposition(sequence, base)
        code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        encodings.append(code)
    return encodings


#---------------

def ENAC(fastas, window=5):
    kw = {'order': 'ACGT'}
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    header = ['#', 'label']
    for w in range(1, len(fastas[0][1]) - window + 2):
        for aa in AA:
            header.append('SW.' + str(w) + '.' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                count = Counter(sequence[j:j + window])
                for key in count:
                    count[key] = count[key] / len(sequence[j:j + window])
                for aa in AA:
                    code.append(count[aa])
        encodings.append(code)
    return encodings

def process_fasta(data_path, temp_file_path):
        # ... code for processing fasta sequences ...
        cnt = 0
        res= open(temp_file_path +'temp.txt','w')
        
        bg= []
        for record in SeqIO.parse(data_path, "fasta"):
            # ids = str(record.id)
            ids = str(record.id).replace("|", "," )
            seq=str(record.seq)
            
            if len(seq) < 41:
              continue
            elif len(seq) == 41 and seq[20]=='C':
              res.write('>'+ids+'_21\n')
              res.write(seq+'\n')
              bg.append(seq[17:24])
            else:
              for i, c in enumerate(seq):
                  if seq[i]=='C':
                     # print(i)
                     left=seq[0:i]
                     # print(left)
                     right = seq[i:]
                     # print(right)
                     if len(left)>=20:
                         left=left[i-20:i]
                         # print(len(left))
                     else:
                          continue
                     if len(right)>=21:
                          right=right[0:21]
                          # print(len(right))
                     else:
                          continue
                     
                     # print(left+right)
                     # print(len(left+right))
                     bg.append(left[-3:]+"C"+right[0:3])
                     #seqs.append(left+right)
                     
                     res.write('>'+ids+'_'+str(i+1)+'\n')
                     res.write(left+right+'\n')
                    
        res.close()     
        # print(bg)
        return bg
    
def predict(data_path,specie):
        data_io = dataProcessing(data_path,"fasta") #path,fileformat
        data_only3 = data_io.reshape(len(data_io),41)
        file=read_nucleotide_sequences(data_path) #ENAC encoded
        cks = CKSNAP(file,gap=5) #gap=5 default
        cc=np.array(cks)
        data_only1 = cc[1:,2:]
        data_only1 = data_only1.astype(np.float)
        ncp=read_nucleotide_sequences(data_path)
        enc, names=NCP(ncp)
        dd=np.array(enc)
        data_only2 = dd[1:,2:]
        data_only2 = data_only2.astype(np.float)
        file=read_nucleotide_sequences(data_path)
        pseeiip = PseEIIP(file)
        pseeiip = np.array(pseeiip)
        data_only6 = pseeiip[1:,2:]
        data_only6 = data_only6.astype(np.float)
        file=read_nucleotide_sequences(data_path)
        enac = ENAC(file)
        enac = np.array(enac)
        data_only9 = enac[1:,2:]
        data_only9 = data_only9.astype(np.float)
        d = np.concatenate((data_only1,data_only2,data_only3,data_only6, data_only9),axis=1)
        data_only=d
        data_only = pd.DataFrame(data_only)
        data_only = data_only.T.reset_index(drop=True).T
        zero_elements_indx = np.load('zero_elements_indx_'+specie+'.npy')
        data_only = data_only.drop(zero_elements_indx,axis=1)
        dataaa=np.array(data_only) 
        loaded_model = pickle.load(open(specie+'.sav', 'rb'))
        result_pred = loaded_model.predict_proba(dataaa)
        return result_pred, names

    
def main(data_path, specie):
    temp_file_path = os.path.dirname(data_path)
    
    # Process fasta sequences
    bg = process_fasta(data_path, temp_file_path)
 
    preds,names = predict((temp_file_path +'temp.txt'),specie)

    # Threshold for classifying sequences
    thresh = 0.5
    
    # Create output
    output = [['Sequence ID', 'Location', 'Sequence Background', 'Probability', 'Result']]
    
    # Iterate over predictions and generate output
    for pred, idx, BG in zip(preds, names, bg):
        if pred[1] >= thresh:
            tmp = [idx.split('_')[0], idx.split('_')[1], BG, str(round(pred[1], 4)), 'm5C Sequence']
        else:
            tmp = [idx.split('_')[0], idx.split('_')[1], BG, str(round(pred[1], 4)), 'Not m5C Sequence']
        output.append(tmp)
    
    # Write output to file
    with open('results.txt', 'w') as f:
        for line in output:
            f.write('{:<25}{:<15}{:<25}{:<15}{:<20}\n'.format(*line))
            
    print("Program run successfully. Please check the results file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='m5C sequence prediction script')
    parser.add_argument('data_path', type=str, help='Path to the data file')
    parser.add_argument('specie', type=str, choices=['human', 'mouse', 'AT', 'drosophila', 'zebrafish'],
                        help='Choose the specie from human, mouse, AT, drosophila, or zebrafish', metavar='specie')
    args = parser.parse_args()

    main(args.data_path, args.specie)
    
    
