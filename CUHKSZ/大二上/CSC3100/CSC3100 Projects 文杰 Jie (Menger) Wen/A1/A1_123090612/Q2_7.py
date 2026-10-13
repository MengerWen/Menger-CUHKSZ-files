# Read input
n_q = input().split()
while len(n_q) < 2:
    n_q += input().split()
n, q = map(int, n_q)
n, q = int(n), int(q)

arr_line = []
while len(arr_line) < n:
    arr_line += input().split()
arr = list(map(int, arr_line))

l_line = []
while len(l_line) < q:
    l_line += input().split()
l_list = list(map(int, l_line))

r_line = []
while len(r_line) < q:
    r_line += input().split()
r_list = list(map(int, r_line))

# Initialize intervals
intervals = {}
intervals[(arr[0], arr[-1])] = (0, n - 1)

valid = True

for i in range(q):
    l_i = l_list[i]
    r_i = r_list[i]
    key = (l_i, r_i)
    if key not in intervals:
        valid = False
        break
    start, end = intervals[key]
    if start >= end:
        valid = False
        break
    del intervals[key]
    m = (start + end) // 2  # Choose any m such that start <= m < end
    if start <= m < end:
        # Left interval
        left_key = (arr[start], arr[m])
        intervals[left_key] = (start, m)
        # Right interval
        right_key = (arr[m + 1], arr[end])
        intervals[right_key] = (m + 1, end)
    else:
        # Should not happen, but added for safety
        valid = False
        break

print(1 if valid else 0)
