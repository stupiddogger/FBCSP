
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mne.decoding import CSP
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from scipy import signal
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random 
random.seed(1)


# In[2]:


#load data
data=scipy.io.loadmat('dataset_BCIcomp1.mat')
data_test=data['x_test']
data_train=data['x_train']
label_train=data['y_train'].reshape(1,-1)-1
label=scipy.io.loadmat('y_test.mat')
label_test=label['y_test'].reshape(1,-1)-1
print(label_test.shape)
print(label_train.shape)
y_train=label_train[0]
y_test=label_test[0]
print(y_train.shape)
print(y_test.shape)
print(data_test.shape)
#bandpass 8Hz-32Hz
b,a=signal.butter(8,[(16/128),(64/128)],'bandpass')
buffer_x_test=signal.filtfilt(b,a,data_test,axis=0)
buffer_x_train=signal.filtfilt(b,a,data_train,axis=0)
print(buffer_x_test.shape)
#change to satisfy shape(n_samples,n_channels,n_features)
#cut the 3.5s--7s EEG signal(3.5*128--7*128)
all_x_train=np.transpose(buffer_x_train,[2,1,0])
all_x_test=np.transpose(buffer_x_test,[2,1,0])
X_train=all_x_train[:,0::2,448:896]
print(X_train.shape)
X_test=all_x_test[:,0::2,448:896]
print(X_test.shape)


# In[3]:

# defination of bandpass filter 
def butter_bandpass(lowcut,highcut,fs,order):
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    b,a==signal.butter(8,[low,high],'bandpass')
    return b,a
def butter_bandpass_filter(data,lowcut,highcut,fs,order):
    b,a=butter_bandpass(lowcut,highcut,fs,order)
    y=signal.filtfilt(b,a,data,axis=2)
    return y


# In[4]:


csp=CSP(n_components=2, reg=None, log=True, norm_trace=False)


# In[5]:

#acquire and combine features of different fequency bands
features_train=[]
features_test=[]
freq=[8,12,16,20,24,28,32]
for freq_count in range(len(freq)):
#loop for freqency
    lower=freq[freq_count]
    if lower==freq[-1]:
        break
    higher=freq[freq_count+1]
    X_train_filt=butter_bandpass_filter(X_train,lowcut=lower,highcut=higher,fs=128,order=8)
    X_test_filt=butter_bandpass_filter(X_test,lowcut=lower,highcut=higher,fs=128,order=8)
    tmp_train=csp.fit_transform(X_train_filt,y_train)
    tmp_test=csp.transform(X_test_filt)
    if freq_count==0:
        features_train=tmp_train
        features_test=tmp_test
    else:
        features_train=np.concatenate((features_train,tmp_train),axis=1)
        features_test=np.concatenate((features_test,tmp_test),axis=1)
print(features_train.shape)
print(features_test.shape)            
    


# In[110]:

#get the best k features base on MIBIF algorithm
select_K=sklearn.feature_selection.SelectKBest(mutual_info_classif,k=10).fit(features_train,y_train)
New_train=select_K.transform(features_train)
#np.random.shuffle(New_train)
New_test=select_K.transform(features_test)
#np.random.shuffle(New_test)
print(New_train.shape)
print(New_test.shape)
ss = preprocessing.StandardScaler()
X_select_train = ss.fit_transform(New_train,y_train)
X_select_test = ss.fit_transform(New_test)


# In[119]:


#calssify
from sklearn.svm import SVC
#from sklearn.grid_search import GridSearchCV
#pipe_svc=([('clf',SVC(random_state=1))])
#param_range=[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100,1000]
#param_grid=[{'clf_C':param_range,
   #         'clf_kernel':['linear']},
  #          {'clf_C':param_range,
 #           'clf_kernel':['rbf']}]
#gs= GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy')
#gs=gs.fit(X_train,y_train)
#print(gs.best_score_)
#print(gs.best_params_)

clf=svm.SVC(C=0.8,kernel='rbf')
clf.fit(X_select_train,y_train)
y_pred=clf.predict(X_select_test)
print(y_test)
print(y_pred)
acc=accuracy_score(y_test,y_pred)
print(acc)

