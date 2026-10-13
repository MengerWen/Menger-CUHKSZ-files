import sys

def main():
    sys.setrecursionlimit(1 << 25)
    n, q = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    l = list(map(int, sys.stdin.readline().split()))
    r = list(map(int, sys.stdin.readline().split()))
    
    pos = [0] * (n +1)
    for idx, val in enumerate(P,1):
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
        key = (a,b)
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
    
    # Check first split is [1,n}
    if not splits:
        print(0)
        return
    first_split = splits[0]
    if first_split != (1,n):
        print(0)
        return
    
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
