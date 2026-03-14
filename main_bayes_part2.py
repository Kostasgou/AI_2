from Bayes import *
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import random
import pandas as pd
import math

train_precisions_bayes = []
train_recalls_bayes = []
train_f1s_bayes = []
dev_precisions_bayes = []
dev_recalls_bayes = []
dev_f1s_bayes = []

train_precisions_gaussian = []
train_recalls_gaussian = []
train_f1s_gaussian = []
dev_precisions_gaussian = []
dev_recalls_gaussian = []
dev_f1s_gaussian = []

TARGET_CLASS = 1

TARGET_CLASS = 1 
def IG(class_, feature):
    classes = set(class_)

    # Calculate overall entropy
    Hc = 0
    for c in classes:
        pc = list(class_).count(c) / len(class_)
        Hc += -pc * math.log(pc, 2)

    feature_values = set(feature)
    Hc_feature = 0

    for feat in feature_values:
        pf = list(feature).count(feat) / len(feature)
        indices = [i for i in range(len(feature)) if feature[i] == feat]
        classes_of_feat = [class_[i] for i in indices]

        for c in classes:
            pcf = classes_of_feat.count(c) / len(classes_of_feat) if len(classes_of_feat) > 0 else 0
            if pcf > 0:
                Hc_feature += -pf * pcf * math.log(pcf, 2)

    ig = Hc - Hc_feature
    return ig


m = 5000
n = 8000
k = 8000
per = 2500

print(f'Number of m: {m}')
print(f'Number of n: {n}')
print(f'Number of k: {k}')  

def split_train_dev(data, dev_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - dev_ratio))
    return data[:split_index], data[split_index:]

sizes = []
for i in range(5):
    print(f'Loop {i+1}:')
   
    train_data_full = sindiasmos_dedomenwn(train_neg, train_pos, per)
    train_data, dev_data = split_train_dev(train_data_full, dev_ratio=0.2)
    test_data = sindiasmos_dedomenwn(test_neg, test_pos, 12500)

    res_train = dimiourgia_lexilogiou(train_data, m, n, k)
    train_vocab = res_train[0]
    train_cat = res_train[1]
    train_texts  = res_train[2]
    
    features=[]
    # Calculate Information Gain for each feature
    for idx, word in enumerate(train_vocab):
        feature_column = []
        for row in train_texts:
            if idx < len(row):  # Έλεγχος ότι το index είναι έγκυρο
                feature_column.append(row[idx])
            else:
                feature_column.append("")  # Προσθήκη κενής τιμής αν δεν υπάρχει το index
        ig = IG(train_cat, feature_column)
        features.append((word, ig))
    
    # Select top m features based on Information Gain
    sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
    train_vocab = [word for word, _ in sorted_features[:m]]
    
    res_dev = dimiourgia_lexilogiou(dev_data, m, n, k)
    dev_vocab = res_dev[0]
    dev_cat = res_dev[1]
    dev_texts  = res_dev[2]
    
    res_test = dimiourgia_lexilogiou(test_data, m, n, k)
    test_vocab = res_test[0]
    test_cat = res_test[1]
    test_texts = res_test[2]
    
    vectorizer = CountVectorizer(vocabulary=train_vocab, binary=True)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_dev = vectorizer.transform(dev_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    # GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, train_cat)
    
    train_predictions_gaussian = nb.predict(X_train)
    dev_predictions_gaussian = nb.predict(X_dev)
    test_predictions_gaussian = nb.predict(X_test)
    

    # Bayes Implementation
    bayes = Bayes()
    bayes.train(X_train, train_cat)
    train_predictions_bayes = bayes.provlepsi(X_train)
    dev_predictions_bayes = bayes.provlepsi(X_dev)
    
    # Metrics for GaussianNB
    train_precision_g, train_recall_g, train_f1_g, _ = precision_recall_fscore_support(train_cat, train_predictions_gaussian, average=None)
    dev_precision_g, dev_recall_g, dev_f1_g, _ = precision_recall_fscore_support(dev_cat, dev_predictions_gaussian, average=None)
    
    train_precisions_gaussian.append(train_precision_g[TARGET_CLASS])
    train_recalls_gaussian.append(train_recall_g[TARGET_CLASS])
    train_f1s_gaussian.append(train_f1_g[TARGET_CLASS])
    dev_precisions_gaussian.append(dev_precision_g[TARGET_CLASS])
    dev_recalls_gaussian.append(dev_recall_g[TARGET_CLASS])
    dev_f1s_gaussian.append(dev_f1_g[TARGET_CLASS])
    
    # Metrics for Bayes Implementation
    train_precision_b, train_recall_b, train_f1_b, _ = precision_recall_fscore_support(train_cat, train_predictions_bayes, average=None)
    dev_precision_b, dev_recall_b, dev_f1_b, _ = precision_recall_fscore_support(dev_cat, dev_predictions_bayes, average=None)
    
    train_precisions_bayes.append(train_precision_b[TARGET_CLASS])
    train_recalls_bayes.append(train_recall_b[TARGET_CLASS])
    train_f1s_bayes.append(train_f1_b[TARGET_CLASS])
    dev_precisions_bayes.append(dev_precision_b[TARGET_CLASS])
    dev_recalls_bayes.append(dev_recall_b[TARGET_CLASS])
    dev_f1s_bayes.append(dev_f1_b[TARGET_CLASS])
    
    sizes.append(per)
    # Classification Reports for GaussianNB
    print("Classification Report - Training Set (GaussianNB):")
    print(classification_report(train_cat, train_predictions_gaussian, zero_division=1))
    print("Classification Report - Testing Set (GaussianNB):")
    print(classification_report(test_cat, test_predictions_gaussian, zero_division=1))

    per += 2500

# Plot Precision
plt.figure()
plt.plot(sizes, train_precisions_gaussian, label="Custom Bayes Train Precision", linestyle='dashed')
plt.plot(sizes, dev_precisions_gaussian, label="Custom Bayes Dev Precision", linestyle='dashed')
plt.plot(sizes, train_precisions_bayes, label="GaussianNB Train Precision")
plt.plot(sizes, dev_precisions_bayes, label="GaussianNB Dev Precision")
plt.xlabel("Training Set Size")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision Comparison for Category:Negative")
plt.show()

# Plot Recall
plt.figure()
plt.plot(sizes, train_recalls_gaussian, label="Custom Bayes Train Recall", linestyle='dashed')
plt.plot(sizes, dev_recalls_gaussian, label="Custom Bayes Dev Recall", linestyle='dashed')
plt.plot(sizes, train_recalls_bayes, label="GaussianNB Train Recall")
plt.plot(sizes, dev_recalls_bayes, label="GaussianNB Dev Recall")
plt.xlabel("Training Set Size")
plt.ylabel("Recall")
plt.legend()
plt.title("Recall Comparison for Category:Negative")
plt.show()

# Plot F1 Score
plt.figure()
plt.plot(sizes, train_f1s_gaussian, label="Custom Bayes Train F1", linestyle='dashed')
plt.plot(sizes, dev_f1s_gaussian, label="Custom Bayes Dev F1", linestyle='dashed')
plt.plot(sizes, train_f1s_bayes, label="GaussianNB Bayes Train F1")
plt.plot(sizes, dev_f1s_bayes, label="GaussianNB Dev F1")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 Score Comparison for Category:Negative")
plt.show()

# Αποτελέσματα στα δεδομένα αξιολόγησης
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_cat, test_predictions_gaussian, average=None)
macro_avg = precision_recall_fscore_support(test_cat, test_predictions_gaussian, average='macro')
micro_avg = precision_recall_fscore_support(test_cat, test_predictions_gaussian, average='micro')

print("Test Data Results:")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1: {test_f1}")
print(f"Macro Average: {macro_avg}")
print(f"Micro Average: {micro_avg}")
