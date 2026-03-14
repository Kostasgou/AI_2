from collections import Counter
import os
import re
import numpy as np
import random
from tqdm import tqdm

class Logistic_Regression:
    def __init__(self, learning_rate=0.001, num_iterations=1000, regularization_param=1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in tqdm(range(self.num_iterations)):
            error = y - self.sigmoid(np.dot(X, self.weights))
            gradient = np.dot(X.T, error) - self.regularization_param * self.weights
            self.weights += self.learning_rate * gradient

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model has not been trained yet.")

        predictions = np.round(self.sigmoid(np.dot(X, self.weights)))
        return predictions.astype(int)

# method for sorting files
pattern = re.compile(r'(\d+)_\d+')


# method for sorting files
pattern = re.compile(r'(\d+)_\d+')

def extract_numerical_part(file_name):
    match = pattern.match(file_name)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number for files without a match

def read_txt_file(folder_path, category, filename):
    data = []
    folder_path = os.path.join(folder_path, filename)
    with open(folder_path, 'r', encoding='utf-8') as file:
        content = file.read()
        data.append(category)   # #1 category
        data.append(filename)   # #2 filename
        data.append(content)    # #3 content
    return data

# combine training data
def combine_data(path_neg, path_pos, per):
    positive_data = []
    negative_data = []
    all_data = []

    neg_files = os.listdir(path_neg)
    pos_files = os.listdir(path_pos)
    # Sort files based on the numerical part
    neg_files.sort(key = extract_numerical_part)
    pos_files.sort(key = extract_numerical_part)

    #limit = (per/100)*25000
    #limit = int((per / 100) * 25000)

    for filename in neg_files[:per]:
        negative_data.append(read_txt_file(path_neg, 1, filename)) # 1 for negatives
    for filename in pos_files[:per]:
        positive_data.append(read_txt_file(path_pos, 0, filename)) # 0 for positives
    # Combine positive and negative data
    all_data = positive_data + negative_data
    #print("-----------combine data-------------")
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
    word_counts = Counter(all_words)    # vriskei syxnothta lekseon
    # sort voc me bash syxnothta - data
    sorted_vocab = sorted(word_counts, key = word_counts.get, reverse=True) # reverse = true wste na einai se fthinoysa seira syxnothtas - apo pio syxnes -> ligotero syxnes
    sorted_vocab = [word.lower() for word in sorted_vocab] # ola peza
    #print(f'Sorted vocab: {len(sorted_vocab)}') 

    #train vocabulary
    vocab = sorted_vocab[n:]
    vocab = list(set(vocab))  # Αφαιρούμε τις διπλότυπες λέξεις
    vocab = sorted(vocab, key=lambda x: word_counts.get(x, 0), reverse=False)
    #vocab = [word.lower() for word in vocab] 
    vocab = vocab[k:]
    #vocab = sorted(word_counts, key = word_counts.get, reverse=True) 
    #vocab = [word.lower() for word in vocab]
    vocab = sorted(vocab, key=lambda x: word_counts.get(x, 0), reverse = True)
    vocab = vocab[0:m]
    res.append(vocab)
    res.append(categories)
    res.append(texts)
    #print("-------------make vocab------------")
    return res