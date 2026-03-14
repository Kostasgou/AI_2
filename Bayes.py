import random
import os
import re
import numpy as np
import pandas as pd
from collections import Counter


train_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\neg"
train_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\pos"
test_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\neg"
test_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\pos"

train_pos = train_pos_path
train_neg = train_neg_path
test_pos = test_pos_path
test_neg = test_neg_path
class Bayes:
    def __init__(self):
        self.pithanotites_katigoriwn = {}
        self.pithanotites_xaraktiristikwn = {}
    
    def train(self,X,Y):
        katigories, counts = np.unique(Y,return_counts = type)
        synolika_deigmata = len(Y)
        for c, count in zip(katigories, counts):
            self.pithanotites_katigoriwn[c] = count / synolika_deigmata
        
        arithmos_xaraktiristikwn = X.shape[1]
        for c in katigories:
            deigma_katigorias = X[Y==c]
            counts_xaraktiristikwn = np.sum(deigma_katigorias, axis=0)
            self.pithanotites_xaraktiristikwn[c] = (counts_xaraktiristikwn+1) / (np.sum(counts_xaraktiristikwn)+arithmos_xaraktiristikwn)#Laplace
    
    def fit(self, X, Y):
        self.train(X, Y)

    


    def provlepsi(self,X):
        provlepsis = []
        for deigma in X:
            max_pithanotita = -np.inf
            provlepomeni_katigoria = None
            for c,pithanotites_katigoriwn in self.pithanotites_katigoriwn.items():
                logarithmiki_pithanotita = np.log(pithanotites_katigoriwn) + np.sum(np.log(self.pithanotites_xaraktiristikwn[c]+ 1e-10)* deigma)
                if logarithmiki_pithanotita>max_pithanotita:
                    max_pithanotita = logarithmiki_pithanotita
                    provlepomeni_katigoria = c
            provlepsis.append(provlepomeni_katigoria)
        return np.array(provlepsis)
    
    def predict(self, X):
        return self.provlepsi(X)
    
# method for sorting files
pattern = re.compile(r'(\d+)_\d+')


def eksagwgi_arithmwn(onoma_arxeiou):
    tairiasma = pattern.match(onoma_arxeiou)
    if tairiasma:
        return int(tairiasma.group(1))
    return float('inf') # epistrefei enan megalo arithmo gia arxeia xwris antistoixeia.

def diavasma_arxeiou(diadromi_fakelou, katigoria, onomaarxeiou):
    dedomena = []
    diadromi_fakelou = os.path.join(diadromi_fakelou, onomaarxeiou)
    with open(diadromi_fakelou, 'r', encoding='utf-8') as arxeio:
        periexomeno = arxeio.read()
        dedomena.append(katigoria)   
        dedomena.append(onomaarxeiou)   
        dedomena.append(periexomeno)    
    return dedomena

# sindiasmos dedomenwn ekpedeyseis
def sindiasmos_dedomenwn(arnitika_diaromi, thetika_diadromi, arithmos_dedomenwn):
    thetika_dedomena = []
    arnitika_dedomena = []
    all_dedomena = []

    arnitika_arxeia = os.listdir(arnitika_diaromi)
    thetika_arxeia = os.listdir(thetika_diadromi)
    arnitika_arxeia.sort(key = eksagwgi_arithmwn)
    thetika_arxeia.sort(key = eksagwgi_arithmwn)


    for onomaarxeiou in arnitika_arxeia[:arithmos_dedomenwn]:
        arnitika_dedomena.append(diavasma_arxeiou(arnitika_diaromi, 1, onomaarxeiou)) # 1 gia arnitika 
    for onomaarxeiou in thetika_arxeia[:arithmos_dedomenwn]:
        thetika_dedomena.append(diavasma_arxeiou(thetika_diadromi, 0, onomaarxeiou)) # 0 gia thetika 
    all_dedomena = thetika_dedomena + arnitika_dedomena
    return all_dedomena

def dimiourgia_lexilogiou(dedomena, m, n, k):
    res = []
    random.shuffle(dedomena)
    keimena = []    # list me ola ta keimena
    katigories = []      # list me categories antistoixa me ta keimena - 0 or 1
    for x in dedomena:
        keimena.append(x[2])
        katigories.append(x[0]) 
    # xwrismos dedeomenon ekpedeusis
    all_lexeis = ' '.join(keimena).split() 
    arithmos_lexewn = Counter(all_lexeis)    #  syxnothta lekseon
    # taxinomisi lexilogiou 
    taxinomimeno_lexilogio = sorted(arithmos_lexewn, key = arithmos_lexewn.get, reverse=True) # reverse = true wste na einai se fthinoysa seira syxnothtas - apo pio syxnes -> ligotero syxnes
    taxinomimeno_lexilogio = [lexi.lower() for lexi in taxinomimeno_lexilogio] # ola peza
    
    lexilogio = taxinomimeno_lexilogio[n:]
    lexilogio = list(set(lexilogio)) # afairesh diplo
    lexilogio = sorted(lexilogio, key=lambda x: arithmos_lexewn.get(x, 0), reverse=False)
    lexilogio = lexilogio[k:]
    lexilogio = sorted(lexilogio, key=lambda x: arithmos_lexewn.get(x, 0), reverse = True)
    lexilogio = lexilogio[0:m]
    res.append(lexilogio)
    res.append(katigories)
    res.append(keimena)
    return res