import json
from verify import display_results

results = []
with open('preliminary_results/one_uav/results.json') as f:
    results = json.load(f)

for n, res in enumerate(results):
    display_results(res, n, save_path='preliminary_results/one_uav/')
