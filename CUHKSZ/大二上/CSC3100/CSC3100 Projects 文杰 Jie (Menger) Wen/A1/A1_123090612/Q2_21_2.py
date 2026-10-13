def main():
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx

    splits = []
    split_set = set()
    invalid = False

    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
        
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

    # Sort intervals based on your idea
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    # Check if the first interval is the entire array
    if splits[0] != (1, n):
        print(0)
        return

    if q == 2:
        a, b = splits[0]
        c, d = splits[1]
        if a != c and b != d:
            print(0)
            return
    elif q > 2:
        for i in range(q):
            a_i, b_i = splits[i]

            # Check with previous and next interval
            if i > 0 and i < q - 1:
                a_prev, b_prev = splits[i - 1]
                a_next, b_next = splits[i + 1]
                if a_i != a_prev and a_i != a_next and b_i != b_prev and b_i != b_next:
                    print(0)
                    return

    print(1)

if __name__ == "__main__":
    main()
