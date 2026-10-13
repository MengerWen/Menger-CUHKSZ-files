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
            a, b = b, a  # Ensure a <= b
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
    splits.sort(key=lambda x: (x[0], x[1]))
    
    # Check if the first interval is the entire array
    if splits[0] != (1, n):
        print(0)
        return

    valid = True
    q = len(splits)
    for i in range(q):
        a_i, b_i = splits[i]
        found = False

        # Check with previous interval
        if i > 0:
            a_prev, b_prev = splits[i - 1]
            if a_i == a_prev or a_i == b_prev or b_i == a_prev or b_i == b_prev:
                found = True

        # Check with next interval
        if i < q - 1:
            a_next, b_next = splits[i + 1]
            if a_i == a_next or a_i == b_next or b_i == a_next or b_i == b_next:
                found = True

        # If neither condition is met, it's invalid
        if not found:
            valid = False
            break

    print(1 if valid else 0)

if __name__ == "__main__":
    main()
