# Read input
n, q = map(int, input().split())
A = list(map(int, input().split()))
l_seq = list(map(int, input().split()))
r_seq = list(map(int, input().split()))

# Map elements to their positions
pos = {A[i]: i for i in range(n)}

# Initialize split index
i = 0

def process(start, end):
    global i
    if start > end:
        return True
    if i >= q:
        return True
    if A[start] != l_seq[i] or A[end] != r_seq[i]:
        return False
    i += 1
    if start == end:
        return True
    # Split between start and end
    # We need to choose a split point between start and end
    # Since we can split anywhere, we choose start+1
    return process(start, start) and process(start+1, end)

# Start processing from the whole array
if process(0, n-1):
    print(1)
else:
    print(0)
