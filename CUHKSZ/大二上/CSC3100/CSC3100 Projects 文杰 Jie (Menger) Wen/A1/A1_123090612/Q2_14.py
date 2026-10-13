def main():
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx
        
    splits = []  # Stores valid split intervals
    split_set = set()  # For checking uniqueness of intervals
    invalid = False  # Flag to mark invalid input
    
    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
    
        # Ensure the interval is valid
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
    
    # Sort intervals by starting point ascending, ending point descending
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    if not splits:
        print(0)
        return
    
    first_split = splits[0]
    if first_split != (1, n):
        print(0)
        return
    
    # Use a stack and child intervals to check nesting and full coverage
    stack = []
    children = {}  # Map from interval to its list of child intervals
    stack.append(first_split)
    children[first_split] = []
    
    for split in splits[1:]:
        a, b = split
        # Handle the top of the stack to find parent interval of current interval
        while stack and not (a >= stack[-1][0] and b <= stack[-1][1]):
            parent = stack.pop()
            # Check if parent's child intervals satisfy conditions
            if len(children[parent]) == 1:
                pass  # Relaxed the condition here
            elif len(children[parent]) == 2:
                c1, c2 = children[parent]
                if c1[1] + 1 != c2[0]:
                    print(0)
                    return
                if c1[0] != parent[0] or c2[1] != parent[1]:
                    print(0)
                    return
            elif len(children[parent]) > 2:
                print(0)
                return
        if not stack:
            print(0)
            return
        parent = stack[-1]
        if len(children[parent]) >= 2:
            print(0)
            return
        # Add current interval as a child of parent interval
        children[parent].append(split)
        stack.append(split)
        children[split] = []
    
    # Process the remaining stack, check if the last intervals satisfy conditions
    while stack:
        parent = stack.pop()
        if len(children[parent]) == 1:
            pass  # Relaxed the condition here
        elif len(children[parent]) == 2:
            c1, c2 = children[parent]
            if c1[1] + 1 != c2[0]:
                print(0)
                return
            if c1[0] != parent[0] or c2[1] != parent[1]:
                print(0)
                return
        elif len(children[parent]) > 2:
            print(0)
            return
    print(1)

if __name__ == "__main__":
    main()
