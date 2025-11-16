"""
decision_trees.py

Clean, commented implementation of ID3 and C4.5 (gain ratio) decision tree
algorithms for categorical data. Loads `playCricket.csv` from the same
directory, trains both algorithms with 5-fold CV, prints the learned tree
and evaluation metrics.
"""

import math
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def entropy(labels):
    """Compute Shannon entropy for a list of labels."""
    counts = Counter(labels)
    total = len(labels)
    ent = 0.0
    for v in counts.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent


def information_gain(X_col, y):
    """Compute information gain of a categorical feature column."""
    base_entropy = entropy(y)
    total = len(y)
    weighted_entropy = 0.0
    for v in set(X_col):
        subset_y = [y[i] for i in range(total) if X_col[i] == v]
        weighted_entropy += (len(subset_y) / total) * entropy(subset_y)
    return base_entropy - weighted_entropy


def gain_ratio(X_col, y):
    """Compute gain ratio (information gain divided by split info)."""
    ig = information_gain(X_col, y)
    total = len(y)
    value_counts = Counter(X_col)
    split_info = 0.0
    for vcount in value_counts.values():
        p = vcount / total
        split_info -= p * math.log2(p) if p > 0 else 0.0
    return ig / split_info if split_info != 0 else 0.0


def id3(X, y, attributes):
    if len(set(y)) == 1:
        return list(set(y))[0]
    if not attributes:
        return Counter(y).most_common(1)[0][0]

    gains = [information_gain([row[attr] for row in X], y) for attr in attributes]
    best_idx = int(np.argmax(gains))
    best_attr = attributes[best_idx]

    tree = {best_attr: {}}
    values = set(row[best_attr] for row in X)

    for v in values:
        sub_X = [row for row in X if row[best_attr] == v]
        sub_y = [y[i] for i, row in enumerate(X) if row[best_attr] == v]
        if not sub_X:
            tree[best_attr][v] = Counter(y).most_common(1)[0][0]
        else:
            new_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][v] = id3(sub_X, sub_y, new_attrs)

    return tree


def c45(X, y, attributes):
    if len(set(y)) == 1:
        return list(set(y))[0]
    if not attributes:
        return Counter(y).most_common(1)[0][0]

    ratios = [gain_ratio([row[attr] for row in X], y) for attr in attributes]
    best_idx = int(np.argmax(ratios))
    best_attr = attributes[best_idx]

    tree = {best_attr: {}}
    values = set(row[best_attr] for row in X)

    for v in values:
        sub_X = [row for row in X if row[best_attr] == v]
        sub_y = [y[i] for i, row in enumerate(X) if row[best_attr] == v]
        if not sub_X:
            tree[best_attr][v] = Counter(y).most_common(1)[0][0]
        else:
            new_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][v] = c45(sub_X, sub_y, new_attrs)
    return tree


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    branches = tree[attr]
    value = sample.get(attr)

    if value not in branches:
        leaves = []

        def collect_labels(subtree):
            if isinstance(subtree, dict):
                for v in subtree.values():
                    collect_labels(v)
            else:
                leaves.append(subtree)

        collect_labels(branches)
        return Counter(leaves).most_common(1)[0][0]

    return predict(branches[value], sample)


def pretty_print_tree(tree, indent=0):
    if not isinstance(tree, dict):
        print(' ' * indent + f'-> {tree}')
        return
    attr = next(iter(tree))
    print(' ' * indent + f'[{attr}]')
    for val, subtree in tree[attr].items():
        print(' ' * (indent + 2) + f'({val}):')
        pretty_print_tree(subtree, indent + 4)


def evaluate(X, y, algorithm_func, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    X = list(X)
    y = list(y)

    for train_idx, test_idx in kf.split(X):
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        attributes = list(X[0].keys())
        tree = algorithm_func(X_train, y_train, attributes)

        y_pred = [predict(tree, sample) for sample in X_test]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["precision"].append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, average="macro", zero_division=0))

    return {k: round(np.mean(v), 4) for k, v in metrics.items()}


def main():
    df = pd.read_csv('playCricket.csv')
    features = df.drop(["PlayCricket", "Day"], axis=1).columns.tolist()
    X = df.drop(["PlayCricket", "Day"], axis=1).to_dict('records')
    y = df['PlayCricket'].tolist()

    print('\nDataset head:')
    print(df.head())

    id3_tree = id3(X, y, features)
    c45_tree = c45(X, y, features)

    print('\n=== ID3 Learned Tree ===')
    pretty_print_tree(id3_tree)

    print('\n=== C4.5 Learned Tree ===')
    pretty_print_tree(c45_tree)

    print('\nEvaluating with 5-fold cross validation...')
    id3_res = evaluate(X, y, id3)
    c45_res = evaluate(X, y, c45)

    print('\nID3 CV Results:', id3_res)
    print('C4.5 CV Results:', c45_res)


if __name__ == '__main__':
    main()
