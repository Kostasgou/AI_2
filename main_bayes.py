from Bayes import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import math

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


train_precisions = []
train_recalls = []
train_f1s = []
dev_precisions = []
dev_recalls = []
dev_f1s = []

m = 5000
n = 8000
k = 8000
per = 2500

# Χωρισμός δεδομένων εκπαίδευσης/ανάπτυξης
def split_train_dev(data, dev_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - dev_ratio))
    return data[:split_index], data[split_index:]

print(f'Number of m: {m}')
print(f'Number of n: {n}')
print(f'Number of k: {k}')  

sizes = []
for i in range(5):  
    print(" ")
    print(f'loop:{i+1}')
   
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
    test_texts  = res_test[2]
    
    vectorizer = CountVectorizer(vocabulary=train_vocab, binary=True)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_dev = vectorizer.transform(dev_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    bayes = Bayes()
    bayes.train(X_train, train_cat)

    train_predictions = bayes.provlepsi(X_train)
    dev_predictions = bayes.provlepsi(X_dev)
    test_predictions = bayes.provlepsi(X_test)
    
    # Υπολογισμός μετρικών για καμπύλες
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_cat, train_predictions, average=None)###
    dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(dev_cat, dev_predictions, average=None)###
    
    train_precisions.append(train_precision[TARGET_CLASS])###
    train_recalls.append(train_recall[TARGET_CLASS])###
    train_f1s.append(train_f1[TARGET_CLASS])###
    dev_precisions.append(dev_precision[TARGET_CLASS])###
    dev_recalls.append(dev_recall[TARGET_CLASS])###
    dev_f1s.append(dev_f1[TARGET_CLASS])###
    
    sizes.append(per)

    # Εκτύπωση αναφορών
    print("Classification Report - Training Set:")
    print(classification_report(train_cat, train_predictions, zero_division=1))
    print("Classification Report - Testing Set:")
    print(classification_report(test_cat, test_predictions, zero_division=1))
    per += 2500

# Γραφικές παραστάσεις
sizes = [2500 * (i+1) for i in range(5)]
plt.figure()
plt.plot(sizes, train_precisions, label="Train Precision")
plt.plot(sizes, dev_precisions, label="Dev Precision")
plt.xlabel("Training Set Size")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision curve for category:Negative")
plt.show()

plt.figure()
plt.plot(sizes, train_recalls, label="Train Recall")
plt.plot(sizes, dev_recalls, label="Dev Recall")
plt.xlabel("Training Set Size")
plt.ylabel("Recall")
plt.legend()
plt.title("Recall curve for category:Negative")
plt.show()

plt.figure()
plt.plot(sizes, train_f1s, label="Train F1")
plt.plot(sizes, dev_f1s, label="Dev F1")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 curve for category:Negative")
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
