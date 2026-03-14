import random
import os
import re
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

train_negative = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\neg"
train_positive = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\pos"
test_negative = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\neg"
test_positive = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\pos"

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        sample_weights = np.ones(len(X)) / len(X) 

        for x in range(self.n_estimators):
            weak_learner = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
            weak_learner.fit(X, y, sample_weight=sample_weights) # ekpaideush tou weak learner
            predictions = weak_learner.predict(X) # provlepseis
            error = np.sum(sample_weights * (predictions != y)) / np.sum(sample_weights) # upologismos sfalmatos
            model_weight = 0.5 * np.log((1 - error) / max(error, 1e-10)) #upologismos varous tou weak learner
            
            sample_weights *= np.exp(-model_weight * np.array(y).astype(int) * predictions)
            sample_weights /= np.sum(sample_weights)
            
            # vazoume ton weak learner sto model
            self.models.append(weak_learner)
            self.model_weights.append(model_weight)

    def predict(self, X):
        predictions = np.array([model.predict(X) * model_weight for model, model_weight in zip(self.models, self.model_weights)])
        return (np.sum(predictions, axis=0) > 0).astype(int)


# diaxwrismos arxeiwn
pattern = re.compile(r'(\d+)_\d+')

def extract_numerical_part(file_name):
    match = pattern.match(file_name)
    if match:
        return int(match.group(1))
    return float('inf') 

def read_txt_file(folder_path, category, filename):
    data = []
    folder_path = os.path.join(folder_path, filename)
    with open(folder_path, 'r', encoding='utf-8') as file:
        content = file.read()
        data.append(category)
        data.append(filename)
        data.append(content)
    return data

# combine training data
def combine_data(path_neg, path_pos, per):
    positive_data = []
    negative_data = []
    all_data = []

    neg_files = os.listdir(path_neg)
    pos_files = os.listdir(path_pos)
    # taksinomei ta arxeia basei tou numerical part
    neg_files.sort(key = extract_numerical_part)
    pos_files.sort(key = extract_numerical_part)


    for filename in neg_files[:per]:
        negative_data.append(read_txt_file(path_neg, 1, filename)) # 1 for negatives
    for filename in pos_files[:per]:
        positive_data.append(read_txt_file(path_pos, 0, filename)) # 0 for positives
    # sundiasmos arxeiwn 
    all_data = positive_data + negative_data
    return all_data    

def make_vocab(data, m, n, k):
    res = []
    random.shuffle(data)
    texts = []    # list me ola ta keimena
    categories = []      # list me categories antistoixa me ta keimena - 0 or 1
    for x in data:
        texts.append(x[2])
        categories.append(x[0]) 
    # split gia train words
    all_words = ' '.join(texts).split() 
    word_counts = Counter(all_words)    #syxnothta lekseon
    # sort lexilogiou me bash syxnothta - data
    sorted_vocab = sorted(word_counts, key = word_counts.get, reverse=True) # reverse = true wste na einai se fthinoysa seira syxnothtas - apo pio syxnes -> ligotero syxnes
    sorted_vocab = [word.lower() for word in sorted_vocab] # ola peza

    #train vocabulary
    vocab = sorted_vocab[n:]
    vocab = list(set(vocab))  # afairesh diplotupwn  
    vocab = sorted(vocab, key=lambda x: word_counts.get(x, 0), reverse=False)
    vocab = vocab[k:]

    vocab = sorted(vocab, key=lambda x: word_counts.get(x, 0), reverse = True)
    vocab = vocab[0:m]
    res.append(vocab)
    res.append(categories)
    res.append(texts)
    return res

