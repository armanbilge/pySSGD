import sys
import itertools as it
from collections import Counter


STATES = {j: i for i, j in enumerate(('A', 'C', 'G', 'T'))}
def parsePatterns(fn):
    patterns = {}
    c = it.count()
    with open(fn) as f:
        for i, j in it.combinations(map(lambda l: l.strip().split(','), filter(lambda l: not l.isspace(), f)), 2):
            a, A, T, G, C, x = i
            b, _, _, _, _, y = j
            counter = Counter(zip(map(STATES.get, x), map(STATES.get, y)))
            for i, n in enumerate(map(int, (A, C, G, T))):
                counter[(i, i)] += n
            patterns[tuple(sorted((a, b)))] = dict(counter)
    return patterns

for fn in sys.argv[1:]:
    print(parsePatterns(fn))