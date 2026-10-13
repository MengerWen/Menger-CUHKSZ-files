# Read input
n, q = map(int, input().split())
permutation = list(map(int, input().split()))
l_values = list(map(int, input().split()))
r_values = list(map(int, input().split()))

# Create a dictionary to store the index of each element in the permutation
index_map = {num: idx for idx, num in enumerate(permutation)}

# Function to check if the permutation is valid
def is_valid_permutation():
    left_boundary = 0
    right_boundary = n - 1
    
    for l, r in zip(l_values, r_values):
        l_index = index_map[l]
        r_index = index_map[r]
        
        # Check if l is on the left side and r is on the right side
        if l_index < r_index and left_boundary <= l_index <= r_index <= right_boundary:
            # Update boundaries
            left_boundary = l_index
            right_boundary = r_index
        else:
            return False
    
    return True

# Check if the permutation is valid and print the result
result = 1 if is_valid_permutation() else 0
print(result)