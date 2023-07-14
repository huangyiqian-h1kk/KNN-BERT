#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[72]:


def concat_shuffle(filelist):
    f0=filelist[0]
    for f in filelist[1:]:
        f0=pd.concat([f0,f])
    print(f0.shape)
    
    train, test = train_test_split(f0, test_size=0.2, random_state=5, shuffle=True)
    valid, test = train_test_split(test, test_size=0.5, random_state=5, shuffle=True)
    
    flist=[train,test,valid]
    train.to_csv('./CPED_train_shuffled.csv', index=False)
    test.to_csv('./CPED_test_shuffled.csv', index=False)
    valid.to_csv('./CPED_valid_shuffled.csv', index=False)


# In[73]:


def get_file(path):
    #pref='./Chinese_corpus/CPED-main/data/CPED/shuffled/'
    #pref='./Chinese_corpus/CPED-main/data/CPED/'
    pref='./'
    with open(pref+path) as f:
        table=pd.read_csv(f)
    return table


# In[56]:


def get_flag(ind,TABLE):#get the posintion of current line in its context(whether its in the beginning/ending or not)
    row_upper = TABLE[TABLE.TV_ID ==TABLE['TV_ID'][ind]].index.tolist()[0]
    row_lower = TABLE[TABLE.TV_ID ==TABLE['TV_ID'][ind]].index.tolist()[-1]
    if ind-row_upper<5:
        flag=ind-row_upper
    elif row_lower-ind<5:
        flag=10-(row_lower-ind)
    else:
        flag=5

    return flag


# In[57]:


def get_context(ind, flag, TABLE):
    context=''
    conlist=['']*11
    conlist[flag]='<mask>'+':'+TABLE['Utterance'][ind]+'。'
    j=0
    for pos in range(ind-flag,ind+10-flag):
        if conlist[j]=='':
            conlist[j]=TABLE['Speaker'][pos]+':'+TABLE['Utterance'][pos]+'。'
        context+=conlist[j]
        j+=1
    return context


# In[68]:


def process(path, output_name):
    file=get_file(path)
    
    tr=get_file('train_split.csv')['Speaker']
    te=get_file('test_split.csv')['Speaker']
    va=get_file('valid_split.csv')['Speaker']
    name_list=list(tr)+list(te)+list(va)
    name_set=set(name_list)
    to_drop=list()
    
    dic=dict()
    for i in list(name_set):
        if not i in dic.keys():
            dic[i]=name_list.count(i)
            
    
    insuf=set([j for j in dic.keys() if dic[j]<20])
    
    name_set=name_set-insuf-{"其他"}

    
    ls=[[0]*4 for _ in range(file.shape[0])]
    #ls=[[0]*3 for _ in range(file.shape[0])]
    
    tv_list=set(file['TV_ID'])
    tv_num=len(tv_list)
    end=[0]
    candidates=dict()
    
    for i in tv_list:
        end.append(file[file.TV_ID ==i].index.tolist()[-1])
    
    for i in range(file.shape[0]):
        
        if not file['TV_ID'][i] in candidates.keys():
            candidates[file['TV_ID'][i]]=[file['Speaker'][i]]
        else:
            candidates[file['TV_ID'][i]].append(file['Speaker'][i]) 
        if i in end[1:]:
            candidates[file['TV_ID'][i]]=set(candidates[file['TV_ID'][i]])
                
                
        if file['Speaker'][i] in name_set:
            ls[i][0]=file['Speaker'][i]
            ls[i][1]=file['Utterance'][i]
            ls[i][2]=get_context(i,get_flag(i,file),file)
            ls[i][3]=file['TV_ID'][i]
        else:
            to_drop.append(i)

    
    df=pd.DataFrame(ls, columns = [ 'lable', 'sentence1','sentence2', 'TV_id' ])
    #df=pd.DataFrame(ls, columns = [ 'label', 'sentence1','sentence2'])
    
    df.insert(df.shape[1], 'candidates', '')
    
    candidates_keys=list(candidates.keys())#list of tv_dis
    for i in range(len(candidates_keys)):
        for j in range(end[i],end[i+1]+1):
            df['candidates'][j]=candidates[candidates_keys[i]]
        end[i+1]+=1
    
    print(df.shape)
    df=df.drop(to_drop)
    print(df.shape)
    df.to_csv(output_name, index=False)
     
    return ls
    

# In[69]:


a=process('train_split.csv', 'CPED_train.csv')


# In[70]:


b=process('test_split.csv', 'CPED_test.csv')


# In[71]:


c=process('valid_split.csv', 'CPED_valid.csv')


# In[84]:


f1.to_csv('./Chinese_corpus/CPED-main/data/CPED/shuffled/train_split.csv',index=False)
f2.to_csv('./Chinese_corpus/CPED-main/data/CPED/shuffled/test_split.csv',index=False)
f3.to_csv('./Chinese_corpus/CPED-main/data/CPED/shuffled/valid_split.csv',index=False)


# In[74]:


f1=get_file('CPED_train.csv')
f2=get_file('CPED_test.csv')
f3=get_file('CPED_valid.csv')


# In[75]:


concat_shuffle([f1,f2,f3])


# In[46]:


f3.shape

