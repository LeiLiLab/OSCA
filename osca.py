import numpy as np
import random
import argparse
from collections import Counter
import json
from tqdm import tqdm

def pass_at_k(n, c, k):
    if n == 0:
        return 0.0
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calculate_expected_train(strategy, data):
    settings_counter = Counter(strategy)
    settings = data['settings']
    
    total_expectation = 0
    
    for p, problem_idx in enumerate(data['training_instances']):
        fail_prob = 1
        for setting_idx, k in settings_counter.items():
            tot_correct = data['train'][str(setting_idx)][p]
            tot_sample = settings[setting_idx]["runs"]
            prob = tot_correct / tot_sample
            fail_prob *= (1 - prob) ** k
        total_expectation += 1 - fail_prob
    
    return total_expectation / len(data['training_instances'])


def calculate_expected_test(strategy, data, subset="testing_instances"):
    settings_counter = Counter(strategy)
    settings = data['settings']
    
    total_expectation = 0
    for p, problem_idx in enumerate(data[subset]):
        fail_prob = 1
        for setting_idx, k in settings_counter.items():
            tot_correct = data['test'][str(setting_idx)][p]
            tot_sample = settings[setting_idx]["runs"]
            fail_prob *= 1 - pass_at_k(tot_sample, tot_correct, k)
        total_expectation += 1 - fail_prob
    
    return total_expectation / len(data[subset])

def optimize_strategy(num_samples, strategy, data):
    improved = True
    while improved:
        improved = False
        for i in range(num_samples):
            best_expectation = calculate_expected_train(strategy, data)
            for s in range(len(data['settings'])):
                if s != strategy[i]:
                    t = strategy[i]
                    strategy[i] = s
                    new_expectation = calculate_expected_train(strategy, data)
                    if new_expectation <= best_expectation:
                        strategy[i] = t
                    else:
                        best_expectation = new_expectation
                        improved = True
    
    return strategy

def hill_climbing(data, num_samples):
    last_strategy = None
    last_pass_rate = -1
    for attempt in range(1):
        random_strategy = random.choices(range(len(data['settings'])), k=num_samples)
        optimized_strategy = optimize_strategy(num_samples, random_strategy, data)
        test_pass_rate = calculate_expected_test(optimized_strategy, data)
        if test_pass_rate > last_pass_rate:
            last_pass_rate = test_pass_rate
            last_strategy = optimized_strategy
    return last_pass_rate, last_strategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--ks", type=str, required=True)
    args = parser.parse_args()
    filename = args.file
    with open(filename, 'r') as f:
        data = json.load(f)
    ks = list(map(lambda x: int(x.strip()), args.ks.split(',')))
    
    print(data['settings'])
    
    pass_rates = []
    for k in tqdm(ks):
        test_pass_rate, optimized_strategy = hill_climbing(data, k)
        print(f"Test pass rate for {k} samples: {test_pass_rate}")
        print(f"Optimized strategy for {k} samples: {optimized_strategy}")
        print()
        pass_rates.append(test_pass_rate)
        
    print(pass_rates)