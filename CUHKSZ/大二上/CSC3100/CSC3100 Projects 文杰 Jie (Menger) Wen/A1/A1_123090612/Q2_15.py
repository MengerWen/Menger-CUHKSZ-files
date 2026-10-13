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
        
    # Stack and children info to check nesting and full coverage
    stack = []
    children = {}  # Record child intervals for each interval
    stack.append(first_split)
    children[first_split] = []
        
    for split in splits[1:]:
        a, b = split
        # Find the parent interval for the current interval
        while stack and not (a >= stack[-1][0] and b <= stack[-1][1]):
            parent = stack.pop()
            # Check if the children's intervals cover the parent interval fully
            child_intervals = children[parent]
            if not child_intervals:
                continue
            child_intervals.sort()
            merged = []
            for interval in child_intervals:
                if not merged:
                    merged.append(interval)
                else:
                    last = merged[-1]
                    if last[1] >= interval[0] - 1:
                        merged[-1] = (last[0], max(last[1], interval[1]))
                    else:
                        print(0)
                        return
            if merged[0][0] != parent[0] or merged[-1][1] != parent[1]:
                print(0)
                return
        if not stack:
            print(0)
            return
        parent = stack[-1]
        # Add current interval as a child of the parent
        children[parent].append(split)
        stack.append(split)
        children[split] = []
        
    # Process remaining intervals in the stack
    while stack:
        parent = stack.pop()
        child_intervals = children[parent]
        if not child_intervals:
            continue
        child_intervals.sort()
        merged = []
        for interval in child_intervals:
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if last[1] >= interval[0] - 1:
                    merged[-1] = (last[0], max(last[1], interval[1]))
                else:
                    print(0)
                    return
        if merged[0][0] != parent[0] or merged[-1][1] != parent[1]:
            print(0)
            return
    print(1)

if __name__ == "__main__":
    main()
