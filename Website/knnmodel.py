import whois
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score

def modifys(smth):
    r = whois.whois(smth)
    #c3 = 0 if type(r.domain_name)=="<class 'NoneType'>" else 1
    X = [len(smth),len(r.domain_name),0.002267]
    c4 = 0 if type(r.whois_server)=="<class 'NoneType'>" else 1
    c5 = 0 if type(r.emails)=="<class 'NoneType'>" else 1
    X.extend([c4,10.857694,2.743793])
    X.extend([5.273185,X_train['tld_present_params'].mean(),c5])
    X.extend([X_train['time_response'].mean(),X_train['asn_ip'].mean(),X_train['domain_google_index'].mean()])
    X.extend([X_train['url_shortened'].mean(),X_train['qty_char_domain'].mean()])
    return X

    
train=pd.read_csv("newdataset.csv")
train = train.drop('time_domain_expiration', axis=1)
train = train.drop('time_domain_activation', axis=1)
data = train.drop('phishing', axis=1)
label = train['phishing']

X_train, X_val, y_train, y_val = train_test_split(data,label,test_size=0.05, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


#########################################              input

#url = input("Enter URL: ")
#y=modifys(url)
##########################################



#print(X)
#c = X_train.columns
#X2=[[c[i],y[i]] for i in range(14)]
#X3 = pd.Series(dict(X2))

#print(X3)

############################################              prediction
'''response = knn.predict([y])
if response == 1:
    print("Phishing URL Alert!!!")
else:
    print("Genuine URL, Proceed...")'''

