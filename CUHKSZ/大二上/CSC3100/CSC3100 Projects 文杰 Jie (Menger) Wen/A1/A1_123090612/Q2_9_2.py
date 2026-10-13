def main():
    # Read n and q from input
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)  # Position array to store the index of each element in P
    for idx, val in enumerate(P, start=1):
        pos[val] = idx  # Store the position of each value in the pos array
    
    splits = []  # List to store valid splits
    split_set = set()  # Set to ensure uniqueness of splits
    invalid = False  # Flag to indicate if the input is invalid
    
    for i in range(q):
        a = pos[l[i]]  # Get the position of l[i] in P
        b = pos[r[i]]  # Get the position of r[i] in P
        
        # Verify that l_i is the first element of the left split and r_i is the last element of the right split
        if P[a - 1] != l[i] or P[b - 1] != r[i]:
            invalid = True
            break
        
        if a > b:  # Ensure that the split is valid
            invalid = True
            break
        
        key = (a, b)
        if key in split_set:  # Check if the split already exists
            invalid = True
            break
        
        split_set.add(key)  # Add the split to the set
        splits.append(key)  # Add the split to the list
    
    if invalid:
        print(0)
        return
    
    # Sort splits by 'a' ascending and 'b' descending
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    # The first split must cover the entire array
    if not splits: # splits is not empty
        print(0)
        return
    
    first_split = splits[0]
    if first_split != (1, n):
        print(0)
        return
    
    # Initialize available splits with the first split
    available_splits = {(1, n)}
    
    for split in splits:
        if split not in available_splits:
            invalid = True
            break
        available_splits.remove(split)  # Remove the processed split
        
        a, b = split
        k = b - 1  # Define the split point k (the point to divide the current segment)
        
        if k < a:
            invalid = True
            break
        
        # Define the two new splits
        left_split = (a, k)
        right_split = (k + 1, b)
        
        # Add the new splits to the available set if they are valid
        if left_split[0] <= left_split[1]:
            available_splits.add(left_split)
        if right_split[0] <= right_split[1]:
            available_splits.add(right_split)
    
    if invalid:
        print(0)
    else:
        # After processing all splits, ensure that no invalid splits remain
        print(1)

if __name__ == "__main__":
    main()
