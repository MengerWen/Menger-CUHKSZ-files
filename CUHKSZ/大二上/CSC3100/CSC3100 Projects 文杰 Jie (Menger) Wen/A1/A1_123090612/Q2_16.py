def main():
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx
        
    splits = []  # Valid intervals
    split_set = set()  # For checking unique intervals
    invalid = False  # Flag for invalid input
        
    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
        
        # Ensure valid interval
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
        
    # Sort intervals: left endpoint ascending, right endpoint descending
    splits.sort(key=lambda x: (x[0], -x[1]))
        
    if not splits:
        print(0)
        return
        
    first_split = splits[0]
    if first_split != (1, n):
        print(0)
        return
        
    # Build parent and children relationships
    stack = [first_split]
    children = {first_split: []}
    for split in splits[1:]:
        a, b = split
        # Find the parent interval for the current interval
        while stack and not (a >= stack[-1][0] and b <= stack[-1][1]):
            stack.pop()
        if not stack:
            print(0)
            return
        parent = stack[-1]
        # Check for overlaps with existing siblings
        for sibling in children[parent]:
            if not (b < sibling[0] or a > sibling[1]):
                # Overlaps with a sibling interval
                print(0)
                return
        # Add current interval as a child of the parent
        children[parent].append(split)
        children[split] = []
        stack.append(split)
        
    print(1)
    
if __name__ == "__main__":
    main()
