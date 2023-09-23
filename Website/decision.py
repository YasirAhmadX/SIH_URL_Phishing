#req: whois,google
try:
	from googlesearch import search
except ImportError:
	print("No module named 'google' found")

import whois
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score

# to get the url in required form
def modifys(smth):
    r = whois.whois(smth)
    #c3 = 0 if type(r.domain_name)=="<class 'NoneType'>" else 1
    X = [len(smth),len(r.domain_name),0.002267]
    c4 = 0 if type(r.whois_server)=="<class 'NoneType'>" else 1
    c5 = 0 if type(r.emails)=="<class 'NoneType'>" else 1
    dir_length= 0 if type(r.emails)=="<class 'NoneType'>" else smth.count('\\')
    X.extend([c4,dir_length,2.743793])
    X.extend([5.273185,X_train['tld_present_params'].mean(),c5])
    X.extend([X_train['time_response'].mean(),X_train['asn_ip'].mean(),X_train['domain_google_index'].mean()])
    X.extend([X_train['url_shortened'].mean(),X_train['qty_char_domain'].mean()])

    #print(X)
    return X

# to search in google index
def fval(smth):
    query = "site:" + ('/'.join(smth.split('/')[:3]))

    for j in search(query, tld="co.in", num=10, stop=10, pause=2):
        print(j)

    try:
        j = j
        print("Listed in google search index")
        flag = 0
    except NameError:
        print("Not listed google search index")
        flag = 1
    
    return flag


train=pd.read_csv("models/newdataset.csv")
train = train.drop('time_domain_expiration', axis=1)
train = train.drop('time_domain_activation', axis=1)
data = train.drop('phishing', axis=1)
label = train['phishing']

X_train, X_val, y_train, y_val = train_test_split(data,label,test_size=0.05, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


#########################################              input


#url = input("Enter URL: ")
#mod_url=modifys(url)

##########################################


#c = X_train.columns
#X2=[[c[i],X[i]] for i in range(14)]
#X3 = pd.Series(dict(X2))

#print(X3)




############################################              prediction
#response = dt_model.predict([mod_url])
#Gflag=fval(url)
"""
if response == 1:
    #print("Phishing URL Alert!!!")
    if Gflag:
        print("Phishing URL Alert!!!")
    else:
        print("Website is listed on google search index, but seems sus!")
else:
    if Gflag:
        print("Website is not listed on google search index, but seems genuine!")
    else:
        print("Genuine URL, Proceed...")
###############################################

if Gflag:
    if response == 1:
        print("Phishing URL Alert!!!")
    else:
        print('website not listed in google but seems genuine')
else:
    if response == 1:
        print("Website is listed on google search index, but seems sus!")
    else:
        print("Genuine URL, Proceed...")
    
############################################"""

#import pickle
#pickle.dump(dt_model,open("model.pkl","wb"))