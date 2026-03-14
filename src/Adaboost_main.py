from AdaBoost import *
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import math

#information gain
def IG(class_, feature):
    classes = set(class_)
    Hc = 0
    for c in classes:
        pc = list(class_).count(c) / len(class_)
        Hc += -pc * math.log(pc, 2) if pc > 0 else 0

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

train_precision = []
train_recall = []
train_f1 = []
dev_precision = []
dev_recall = []
dev_f1 = []

test_results = {}
train_data = []
test_data = []

m = 5000
n = 8000
k = 15000
per_train = 2500

# Φόρτωση test δεδομένων
test_data = combine_data(test_negative, test_positive, 12500)
selected_category = 0  # για κατηγορία θετικά

for i in range(5):  
    print(f'Iteration Num: {i+1}')
    print("Percentage of data used:", (per_train/12500)*100)
    
    train_data = combine_data(train_negative, train_positive, per_train)
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Δημιουργία λεξιλογίου
    res = make_vocab(train_data, m, n, k)
    train_vocab = res[0]
    train_cat = res[1]
    train_texts  = res[2]

    # Υπολογισμός Information Gain για κάθε χαρακτηριστικό
    features = []
    for idx, word in enumerate(train_vocab):
        feature_column = []
        for row in train_texts:
            feature_column.append(row.split().count(word))
        ig = IG(train_cat, feature_column)
        features.append((word, ig))
    
    # Ταξινόμηση βάσει IG και επιλογή των top-m
    sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
    train_vocab = [word for word, _ in sorted_features[:m]]

    res = make_vocab(test_data, m, n, k)
    test_vocab = res[0]
    test_cat = res[1]
    test_texts  = res[2]
    
    # Μετατροπή σε διανυσματική μορφή
    vectorizer = CountVectorizer(vocabulary=train_vocab, binary=True)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    
    # Εκπαίδευση AdaBoost
    adaboost = AdaBoost(n_estimators=100)
    adaboost.fit(X_train, train_cat)
    
    # Προβλέψεις
    train_predictions = adaboost.predict(X_train)
    test_predictions = adaboost.predict(X_test)



    
    # Υπολογισμός precision, recall, f1
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(train_cat, train_predictions, average=None)
    precision_dev, recall_dev, f1_dev, _ = precision_recall_fscore_support(test_cat, test_predictions, average=None)
    
    # Αποθήκευση μετρικών
    train_precision.append(precision_train[selected_category])
    train_recall.append(recall_train[selected_category])
    train_f1.append(f1_train[selected_category])
    
    dev_precision.append(precision_dev[selected_category])
    dev_recall.append(recall_dev[selected_category])
    dev_f1.append(f1_dev[selected_category])

    print(f"\nEvaluation on Test Data - Iteration {i+1}:")
    report = classification_report(test_cat, test_predictions, zero_division=1, output_dict=True)

    test_results['Precision'] = report['macro avg']['precision']
    test_results['Recall'] = report['macro avg']['recall']
    test_results['F1-score'] = report['macro avg']['f1-score']
    
    df = pd.DataFrame(report).transpose()
    print(df)
    
    per_train += 2500

sizes = [2500 * (i+1) for i in range(5)]

#sxediash kampulwn
plt.figure()
plt.plot(sizes, train_precision, label='Train Precision', marker='o')
plt.plot(sizes, dev_precision, label='Dev Precision', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('Precision')
plt.title('Learning Curve - Precision')
plt.legend()
plt.show()

plt.figure()
plt.plot(sizes, train_recall, label='Train Recall', marker='o')
plt.plot(sizes, dev_recall, label='Dev Recall', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('Recall')
plt.title('Learning Curve - Recall')
plt.legend()
plt.show()

plt.figure()
plt.plot(sizes, train_f1, label='Train F1-score', marker='o')
plt.plot(sizes, dev_f1, label='Dev F1-score', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('F1-score')
plt.title('Learning Curve - F1-score')
plt.legend()
plt.show()
