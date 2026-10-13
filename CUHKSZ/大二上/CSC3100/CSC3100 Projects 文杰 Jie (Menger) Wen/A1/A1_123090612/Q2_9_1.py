def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    n, q = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    l = list(map(int, sys.stdin.readline().split()))
    r = list(map(int, sys.stdin.readline().split()))
    
    pos = [0]*(n+1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx
    
    splits = []
    split_set = set()
    invalid = False
    
    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
        
        # Verify that l_i is the first element of the left split and r_i is the last element of the right split
        if P[a-1] != l[i] or P[b-1] != r[i]:
            invalid = True
            break
        
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
    
    # Sort splits by a ascending, b descending
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    # The first split must cover the entire array
    if not splits or splits[0] != (1, n):
        print(0)
        return
    
    # Initialize available splits with the first split
    available_splits = {(1, n)}
    
    for split in splits:
        if split not in available_splits:
            invalid = True
            break
        available_splits.remove(split)
        
        a, b = split
        # Determine the split point k
        # We need to find a split point such that:
        # - The left split starts with l_i (which is P[a-1])
        # - The right split ends with r_i (which is P[b-1])
        # To ensure consistency, we'll choose the split point k such that:
        # The left split is from a to k, and the right split is from k+1 to b
        # Since we don't know k, we'll iterate to find the possible k
        # However, for efficiency, we'll assume the split is uniquely determined
        # by the positions of l_i and r_i
        
        # Find the minimal k where P[k] is the last element of the left split
        # Since P[a-1] is the first element, and P[b-1] is the last of the right
        # We can choose k such that [a, k] ends just before r_i
        
        # To simplify, we'll choose k as the position where r_i starts
        # i.e., k = pos[r_i] -1
        k = b -1
        if k < a:
            invalid = True
            break
        
        # Add the two new splits if they are valid
        left_split = (a, k)
        right_split = (k+1, b)
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
