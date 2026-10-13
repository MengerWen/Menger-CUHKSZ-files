n, q = map(int, input().split())
P = list(map(int, input().split()))
l_i = list(map(int, input().split()))
r_i = list(map(int, input().split()))
pos = {}
for idx, val in enumerate(P):
    pos[val] = idx

segments = set()
segments.add((0, n - 1))
from collections import defaultdict, deque

mapping = defaultdict(deque)
mapping[(P[0], P[n - 1])].append((0, n - 1))

valid = True

for idx in range(q):
    l = l_i[idx]
    r = r_i[idx]
    if (l, r) not in mapping or not mapping[(l, r)]:
        valid = False
        break
    found = False
    while mapping[(l, r)]:
        start, end = mapping[(l, r)].popleft()
        if end - start + 1 >= 2:
            found = True
            break
    if not found:
        valid = False
        break
    segments.discard((start, end))
    mid = start  # You can choose any mid between start and end - 1
    left_seg = (start, mid)
    right_seg = (mid + 1, end)
    segments.add(left_seg)
    segments.add(right_seg)
    mapping[(P[left_seg[0]], P[left_seg[1]])].append(left_seg)
    mapping[(P[right_seg[0]], P[right_seg[1]])].append(right_seg)

print(1 if valid else 0)
