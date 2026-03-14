from AdaBoost import *
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd

train_data = []
test_data = []

m = 5000
n = 8500
k = 8500
per = 2500 

sizes = [2500 * (i + 1) for i in range(5)]
algorithms = {
    'Custom AdaBoost': AdaBoost(n_estimators=100),
    'Sklearn AdaBoost': AdaBoostClassifier(n_estimators=100)
}

train_precisions, train_recalls, train_f1s = {}, {}, {}
dev_precisions, dev_recalls, dev_f1s = {}, {}, {}

for name in algorithms.keys():
    train_precisions[name], train_recalls[name], train_f1s[name] = [], [], []
    dev_precisions[name], dev_recalls[name], dev_f1s[name] = [], [], []

for i, size in enumerate(sizes):
    print(f'Iteration Num: {i+1}')
    train_data = combine_data(train_negative, train_positive, size)
    test_data = combine_data(test_negative, test_positive, 12500)

    res = make_vocab(train_data, m, n, k)
    train_vocab, train_cat, train_texts = res
    res = make_vocab(test_data, m, n, k)
    test_vocab, test_cat, test_texts = res
    
    train_cat = np.where(np.array(train_cat) == 0, -1, 1) 
    test_cat = np.where(np.array(test_cat) == 0, -1, 1)
    
    vectorizer = CountVectorizer(vocabulary=train_vocab, binary=True)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    
    iteration_results = []
    
    for name, model in algorithms.items():
        model.fit(X_train, train_cat)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        precision_train = precision_score(train_cat, train_predictions, average='macro', zero_division=1)
        recall_train = recall_score(train_cat, train_predictions, average='macro', zero_division=1)
        f1_train = f1_score(train_cat, train_predictions, average='macro', zero_division=1)
        precision_dev = precision_score(test_cat, test_predictions, average='macro', zero_division=1)
        recall_dev = recall_score(test_cat, test_predictions, average='macro', zero_division=1)
        f1_dev = f1_score(test_cat, test_predictions, average='macro', zero_division=1)

        train_precisions[name].append(precision_train)
        train_recalls[name].append(recall_train)
        train_f1s[name].append(f1_train)
        dev_precisions[name].append(precision_dev)
        dev_recalls[name].append(recall_dev)
        dev_f1s[name].append(f1_dev)
        
        iteration_results.append([name, precision_dev, recall_dev, f1_dev])
    
    per += 2500
    
    df_iteration = pd.DataFrame(iteration_results, columns=['Algorithm', 'Precision', 'Recall', 'F1-score'])
    print(f"Evaluation Table {i+1}:")
    print(df_iteration)

for metric, train_values, dev_values in [('Precision', train_precisions, dev_precisions),
                                         ('Recall', train_recalls, dev_recalls),
                                         ('F1-score', train_f1s, dev_f1s)]:
    plt.figure()
    for name in algorithms.keys():
        plt.plot(sizes, train_values[name], label=f'{name} Train', marker='o')
        plt.plot(sizes, dev_values[name], label=f'{name} Dev', marker='s')
    plt.xlabel('Training Set Size')
    plt.ylabel(metric)
    plt.title(f'Learning Curve - {metric}')
    plt.legend()
    plt.show()