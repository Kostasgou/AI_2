import pandas as pd
from Logistic_Regression import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import math
import random
from collections import Counter

# Ορισμός διαδρομών δεδομένων
train_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\neg"
train_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\pos"
test_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\neg"
test_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\pos"


# Information Gain
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

# Διαχωρισμός δεδομένων
def split_train_dev(data, dev_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - dev_ratio))
    return data[:split_index], data[split_index:]

# Αρχικοποίηση λιστών για αποθήκευση μετρικών

train_precisions, train_recalls, train_f1s = [], [], []
dev_precisions, dev_recalls, dev_f1s = [], [], []  # <-- Νέες λίστες για dev set
test_precisions, test_recalls, test_f1s = [], [], []

m, n, k, per = 5000, 8500, 8500, 2500
sizes = [2500 * (i+1) for i in range(5)]

for i in range(5):  
    print(f'Iteration {i+1}')
    
    train_data = combine_data(train_neg_path, train_pos_path, per)
    test_data = combine_data(test_neg_path, test_pos_path, 12500)
    
    train_data, dev_data = split_train_dev(train_data, dev_ratio=0.2)
    
    train_vocab, train_cat, train_texts = make_vocab(train_data, m, n, k)
    dev_vocab, dev_cat, dev_texts = make_vocab(dev_data, m, n, k)  # <-- Προσθήκη dev
    test_vocab, test_cat, test_texts = make_vocab(test_data, m, n, k)
    
    vectorizer = CountVectorizer(vocabulary=train_vocab, binary=True)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_dev = vectorizer.transform(dev_texts).toarray()  # <-- Μετατροπή dev set
    X_test = vectorizer.transform(test_texts).toarray()
    
    models = {
        "Logistic Regression (sklearn)": LogisticRegression(),
        "Custom Logistic Regression": Logistic_Regression()
    }
    
    for model_name, model in models.items():
        model.fit(X_train, train_cat)
        
        train_predictions = model.predict(X_train)
        dev_predictions = model.predict(X_dev)  # <-- Πρόβλεψη στο dev set
        test_predictions = model.predict(X_test)
        
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_cat, train_predictions, average=None)
        dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(dev_cat, dev_predictions, average=None)  # <-- Υπολογισμός dev
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_cat, test_predictions, average=None)

        train_precisions.append((model_name, train_precision[1]))
        train_recalls.append((model_name, train_recall[1]))
        train_f1s.append((model_name, train_f1[1]))
        
        dev_precisions.append((model_name, dev_precision[1]))  # <-- Αποθήκευση dev
        dev_recalls.append((model_name, dev_recall[1]))
        dev_f1s.append((model_name, dev_f1[1]))

        test_precisions.append((model_name, test_precision[1]))
        test_recalls.append((model_name, test_recall[1]))
        test_f1s.append((model_name, test_f1[1]))

        if model_name == "Logistic Regression (sklearn)":  
            print("Classification Report - Training Set (Logistic Regression):")
            print(classification_report(train_cat, train_predictions, zero_division=1))
            print("Classification Report - Dev Set (Logistic Regression):")  # <-- Εκτύπωση dev
            print(classification_report(dev_cat, dev_predictions, zero_division=1))
            print("Classification Report - Testing Set (Logistic Regression):")
            print(classification_report(test_cat, test_predictions, zero_division=1))
    
    per += 2500

# Σχεδίαση γραφημάτων
metrics = {
    "Precision": (train_precisions, dev_precisions, test_precisions),
    "Recall": (train_recalls, dev_recalls, test_recalls),
    "F1 Score": (train_f1s, dev_f1s, test_f1s)
}

for metric_name, (train_metric, dev_metric, test_metric) in metrics.items():
    plt.figure()
    for model_name in models.keys():
        train_values = [val for name, val in train_metric if name == model_name]
        dev_values = [val for name, val in dev_metric if name == model_name]  # <-- Dev values
        test_values = [val for name, val in test_metric if name == model_name]
        
        plt.plot(sizes, train_values, label=f'Train ({model_name})')
        plt.plot(sizes, dev_values, label=f'Dev ({model_name})', linestyle='dotted')  # <-- Προσθήκη dev στο διάγραμμα
        
    
    plt.xlabel('Training Set Size')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} Comparison')
    plt.show()

print("Test Data Results:")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1: {test_f1}")

