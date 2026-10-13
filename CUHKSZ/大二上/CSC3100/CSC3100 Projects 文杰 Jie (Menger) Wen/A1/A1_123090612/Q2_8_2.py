def main():
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start = 1):
        pos[val] = idx
    
    splits = [] # List to store valid splits
    split_set = set() # Set to ensure uniqueness of splits
    invalid = False # Flag to indicate if the input is invalid
    
    # Enter the main loop of query processing,
    # traversing the left and right boundaries of each operation in turn.
    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]

        # Ensure that the split is valid
        if a > b:
            invalid = True
            break
        key = (a, b)
        if key in split_set:
            invalid = True
            break
        split_set.add(key)
        splits.append(key)
    
    if invalid:
        print(0)
        return
    
    # Sort splits by 'a' ascending and 'b' descending.
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    # Check first split is [1,n]
    if not splits: # splits is not empty
        print(0)
        return
    
    first_split = splits[0]
    if first_split != (1, n):
        print(0)
        return
    
    # Check nested relationships using the stack.
    stack = []
    stack.append(first_split)
    
    for split in splits[1:]:
        a, b = split
        top_a, top_b = stack[-1]
        if a > top_b:
            stack.append(split)
        elif a >= top_a and b <= top_b:
            stack.append(split)
        else:
            print(0)
            return
    print(1)

if __name__ == "__main__":
    main()
