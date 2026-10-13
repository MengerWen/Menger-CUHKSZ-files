def main():

    n, q = map(int, input().split())

    P = list(map(int, input().split()))

    l = list(map(int, input().split()))

    r = list(map(int, input().split()))

    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx

    splits = []  # Store valid split intervals
    split_set = set()  # To check for unique intervals
    invalid = False  # Flag to mark if input is invalid

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

    # Sort by left endpoint ascending, right endpoint descending
    splits.sort(key=lambda x: (x[0], -x[1]))
    if not splits:
        print(0)
        return

    first_split = splits[0]
    if first_split != (1, n):
        print(0)
        return

    # Use stack and subinterval info to check nesting and full coverage
    stack = []
    children = {}  # Record child intervals for each interval

    stack.append(first_split)
    children[first_split] = []

    for split in splits[1:]:
        a, b = split
        # Adjust stack to find the current interval's parent
        while stack and not (a >= stack[-1][0] and b <= stack[-1][1]):
            parent = stack.pop()
            # Check if parent interval's child intervals are valid
            if len(children[parent]) == 1:
                child_interval = children[parent][0]
                # Get parent interval's endpoints
                p_left, p_right = parent
                # Get child interval's endpoints
                c_left, c_right = child_interval
                # Ensure child interval is strictly inside parent
                if not (p_left < c_left and c_right < p_right):
                    print(0)
                    return
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

        # Add current interval as a child of the parent interval
        children[parent].append(split)
        stack.append(split)
        children[split] = []

    # Process remaining intervals in the stack
    while stack:
        parent = stack.pop()
        if len(children[parent]) == 1:
            child_interval = children[parent][0]
            p_left, p_right = parent
            c_left, c_right = child_interval
            if not (p_left < c_left and c_right < p_right):
                print(0)
                return
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
