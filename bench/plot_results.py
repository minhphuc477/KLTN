"""Plot benchmark results from CSV produced by bench/suite.py"""
import csv
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_csv(path='bench/results.csv'):
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['time_sec'] = float(r['time_sec'])
            r['nodes_expanded'] = int(r['nodes_expanded'])
            r['path_len'] = int(r['path_len'])
            r['allow_diagonal'] = r['allow_diagonal'] == 'True'
            data.append(r)

    # group by map_type and diag
    groups = defaultdict(list)
    for r in data:
        groups[(r['map_type'], r['allow_diagonal'])].append(r)

    for (map_type, diag), rows in groups.items():
        plt.figure(figsize=(8,4))
        solvers = list(set(r['solver'] for r in rows))
        for s in solvers:
            rs = [r for r in rows if r['solver']==s]
            nodes = [r['nodes_expanded'] for r in rs]
            times = [r['time_sec'] for r in rs]
            plt.scatter(nodes, times, label=s)
        plt.xlabel('nodes_expanded')
        plt.ylabel('time_sec')
        plt.title(f'{map_type} diag={diag}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    plot_csv()