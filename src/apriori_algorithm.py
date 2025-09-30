import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter, defaultdict

class AprioriAlgorithm:
    def __init__(self, min_support=0.01, min_confidence=0.1):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def get_transactions(self, df, group_by='order_id', item_col='dish_name'):
        transactions = df.groupby(group_by)[item_col].apply(list).tolist()
        return transactions

    def get_frequent_1_itemsets(self, transactions):
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        total = len(transactions)
        return {frozenset([item]): count / total for item, count in item_counts.items()
                if count / total >= self.min_support}

    def get_frequent_k_itemsets(self, transactions, prev_frequent, k):
        candidates = set()
        prev_items = list(prev_frequent.keys())
        for i in range(len(prev_items)):
            for j in range(i + 1, len(prev_items)):
                union = prev_items[i] | prev_items[j]
                if len(union) == k:
                    candidates.add(union)
        counts = defaultdict(int)
        for transaction in transactions:
            transaction = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counts[candidate] += 1
        total = len(transactions)
        return {itemset: count / total for itemset, count in counts.items() if count / total >= self.min_support}

    def run(self, transactions):
        frequent = dict()
        k = 1
        current = self.get_frequent_1_itemsets(transactions)
        while current:
            frequent.update(current)
            k += 1
            current = self.get_frequent_k_itemsets(transactions, current, k)
        return frequent

    def generate_rules(self, frequent_itemsets):
        rules = []
        for itemset in frequent_itemsets:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_sup = frequent_itemsets.get(antecedent, 0)
                    consequent_sup = frequent_itemsets.get(consequent, 0)
                    union_sup = frequent_itemsets[itemset]
                    if antecedent_sup > 0:
                        confidence = union_sup / antecedent_sup
                        if confidence >= self.min_confidence:
                            lift = 0
                            if consequent_sup > 0:
                                lift = confidence / consequent_sup
                            rules.append({
                                'antecedents': list(antecedent),
                                'consequents': list(consequent),
                                'support': union_sup,
                                'confidence': confidence,
                                'lift': lift
                            })
        return rules
