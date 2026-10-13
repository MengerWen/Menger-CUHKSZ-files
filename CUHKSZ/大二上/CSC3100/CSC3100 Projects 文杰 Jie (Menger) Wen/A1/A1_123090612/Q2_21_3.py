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
        
        # 如果a > b，则交换a和b，确保有效区间
        if a > b:
            a, b = b, a

        key = (a, b)

        if key in split_set:
            invalid = True
            break

        split_set.add(key)
        splits.append(key)

    if invalid:
        print(0)
        return

    # 按照a升序，b降序排序
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    # 检查第一个分割是否是从1到n（覆盖整个数组）
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

            # 与前一个和后一个区间比较
            if i > 0 and i < q - 1:
                a_prev, b_prev = splits[i - 1]
                a_next, b_next = splits[i + 1]
                if a_i != a_prev and a_i != a_next and b_i != b_prev and b_i != b_next:
                    print(0)
                    return

    print(1)

if __name__ == "__main__":
    main()
