import os
import random
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from Logistic_Regression import *
import pickle
import pandas as pd

# Χώροι για δεδομένα
train_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\neg"
train_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\pos"
test_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\neg"
test_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\pos"

# Συνάρτηση για να βρούμε το αριθμητικό μέρος του ονόματος του αρχείου
def extract_numerical_part(file_name):
    pattern = re.compile(r'(\d+)_\d+')
    match = pattern.match(file_name)
    return int(match.group(1)) if match else float('inf')

# Ανάγνωση των αρχείων κειμένων
def read_txt_file(folder_path, category, filename):
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
        return [category, filename, file.read()]

# Συνδυασμός δεδομένων από αρχεία
def combine_data(path_neg, path_pos, per):
    neg_files = sorted([entry.name for entry in os.scandir(path_neg) if entry.is_file()], key=extract_numerical_part)[:per]
    pos_files = sorted([entry.name for entry in os.scandir(path_pos) if entry.is_file()], key=extract_numerical_part)[:per]
    return [read_txt_file(path_neg, 1, f) for f in neg_files] + [read_txt_file(path_pos, 0, f) for f in pos_files]

# Υπολογισμός Information Gain (IG)
def IG(class_, feature):
    classes = set(class_)
    Hc = sum(-pc * math.log(pc, 2) for c in classes if (pc := class_.count(c) / len(class_)))
    feature_values = set(feature)
    Hc_feature = sum(-pf * (pcf * math.log(pcf, 2) if pcf > 0 else 0)
                      for feat in feature_values
                      if (pf := feature.count(feat) / len(feature))
                      for c in classes
                      if (pcf := sum(1 for i in range(len(feature)) if feature[i] == feat and class_[i] == c) / feature.count(feat)))
    return Hc - Hc_feature

# Δημιουργία λεξιλογίου από τα δεδομένα
def make_vocab(data, m, n, k):
    random.shuffle(data)
    texts, categories = zip(*[(x[2].lower(), x[0]) for x in data])
    
    word_counts = Counter(' '.join(texts).split())
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab = sorted_vocab[max(0, n-500):]
    vocab = list(set(vocab))
    vocab = sorted(vocab, key=lambda x: word_counts.get(x, 0), reverse=True)[max(0, min(k, 10000)-10000):min(m, 5000)]
    
    features = []
    for word in vocab:
        feature_column = [1 if word in text else 0 for text in texts]
        ig = IG(categories, feature_column)
        features.append((word, ig))
    
    sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
    
    num_features = max(m // 2, min(m, len(sorted_features)))
    if num_features == 0:
        print("Warning: IG filtering removed all words. Using top frequent words instead.")
        return sorted_vocab[:m], categories, texts
    
    vocab = [word for word, _ in sorted_features[:num_features]]
    return vocab, categories, texts

# Διαχωρισμός δεδομένων σε εκπαίδευση και ανάπτυξη
def split_train_dev(data, dev_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - dev_ratio))
    return data[:split_index], data[split_index:]

# Βασικά αποτελέσματα και μετρικές
train_precisions, train_recalls, train_f1s = [], [], []
dev_precisions, dev_recalls, dev_f1s = [], [], []
test_precisions, test_recalls, test_f1s = [], [], []

m, n, k, per = 5000, 500, 10000, 2500
sizes = [2500 * (i+1) for i in range(5)]

for i in range(5):
    print(f'Iteration {i+1}')
    
    # Διαβάζουμε τα δεδομένα
    train_data = combine_data(train_neg_path, train_pos_path, per)
    test_data = combine_data(test_neg_path, test_pos_path, 12500)

    # Διαχωρισμός δεδομένων σε εκπαίδευση και ανάπτυξη
    train_data, dev_data = split_train_dev(train_data, dev_ratio=0.2)
    
    # Δημιουργία λεξιλογίου
    train_vocab, train_cat, train_texts = make_vocab(train_data, m, n, k)
    dev_vocab, dev_cat, dev_texts = make_vocab(dev_data, m, n, k)
    test_vocab, test_cat, test_texts = make_vocab(test_data, m, n, k)
    
    # Δοκιμάζουμε με Logistic Regression
    vectorizer = CountVectorizer(vocabulary=train_vocab, binary=True)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_dev = vectorizer.transform(dev_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    model = Logistic_Regression()
    model.fit(X_train, train_cat)
    
    # Κάνουμε προβλέψεις
    train_predictions = model.predict(X_train)
    dev_predictions = model.predict(X_dev)
    test_predictions = model.predict(X_test)
    
    # Αξιολόγηση των αποτελεσμάτων
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_cat, train_predictions, average=None)
    dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(dev_cat, dev_predictions, average=None)
    
    train_precisions.append(train_precision[1])
    train_recalls.append(train_recall[1])
    train_f1s.append(train_f1[1])
    dev_precisions.append(dev_precision[1])
    dev_recalls.append(dev_recall[1])
    dev_f1s.append(dev_f1[1])
    
    # Εκτύπωση αναφορών
    print("Classification Report - Training Set:")
    print(classification_report(train_cat, train_predictions))
    print("Classification Report - Development Set:")
    print(classification_report(dev_cat, dev_predictions))
    print("Classification Report - Test Set:")
    print(classification_report(test_cat, test_predictions))
    
    per += 2500

# Σχεδίαση γραφημάτων μετρικών
sizes = [2500 * (i+1) for i in range(5)]
plt.figure()
plt.plot(sizes, train_precisions, label="Train Precision")
plt.plot(sizes, dev_precisions, label="Dev Precision")
plt.xlabel("Training Set Size")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision curve for category:" )###
plt.show()

plt.figure()
plt.plot(sizes, train_recalls, label="Train Recall")
plt.plot(sizes, dev_recalls, label="Dev Recall")
plt.xlabel("Training Set Size")
plt.ylabel("Recall")
plt.legend()
plt.title("Recall curve for category:" )###
plt.show()

plt.figure()
plt.plot(sizes, train_f1s, label="Train F1")
plt.plot(sizes, dev_f1s, label="Dev F1")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 curve for category:" )###
plt.show()

# Αποτελέσματα στα δεδομένα αξιολόγησης
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_cat, test_predictions, average=None)
macro_avg = precision_recall_fscore_support(test_cat, test_predictions, average='macro')
micro_avg = precision_recall_fscore_support(test_cat, test_predictions, average='micro')

print("Test Data Results:")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1: {test_f1}")
print(f"Macro Average: {macro_avg}")
print(f"Micro Average: {micro_avg}")