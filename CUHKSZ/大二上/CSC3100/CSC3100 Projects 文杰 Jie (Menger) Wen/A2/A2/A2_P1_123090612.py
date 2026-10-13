# Read input
n = int(input())
A = list(map(int, input().split()))

# Step 1: Count occurrences of each element in A
counts = {}
for a in A:
    counts[a] = counts.get(a, 0) + 1

# Initialize the set of all positions
positions = set(range(1, n + 1))

# Step 2: Identify positions to delete initially
positions_to_remove = []
for pos in positions:
    if pos not in counts:
        positions_to_remove.append(pos)

# Step 3: Use a queue to process deletions
queue = positions_to_remove.copy()
deletions = set()

while queue:
    pos = queue.pop(0)  # Using list as a queue
    if pos in deletions:
        continue
    deletions.add(pos)
    
    # Decrease the count of the corresponding element in A
    elem = A[pos - 1]  # Adjusting for zero-based index
    counts[elem] -= 1
    
    if counts[elem] == 0:
        del counts[elem]
        # If the element corresponds to a position, we may need to delete that position
        if elem in positions and elem not in deletions:
            queue.append(elem)

# Step 4: Output the minimum number of deletions
print(len(deletions))
