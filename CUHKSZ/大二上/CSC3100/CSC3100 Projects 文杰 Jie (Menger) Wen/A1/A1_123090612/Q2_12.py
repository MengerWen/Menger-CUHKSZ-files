def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx

    intervals = {}  # 存储区间的父子关系
    intervals[(1, n)] = []  # 初始区间为整个数组

    splits = []
    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
        if a > b:
            print(0)
            return
        splits.append((a, b))

    # 处理分割操作
    used_intervals = set()
    def find_parent(a, b, node):
        # 检查 [a, b] 是否在当前节点内，若是则返回节点
        start, end = node
        if a >= start and b <= end:
            # 如果该节点已被分割，则在子节点中继续查找
            if node in intervals and intervals[node]:
                for child in intervals[node]:
                    res = find_parent(a, b, child)
                    if res:
                        return res
                return None
            else:
                return node
        else:
            return None

    for a, b in splits:
        parent = find_parent(a, b, (1, n))
        if not parent:
            print(0)
            return
        # 检查父区间是否已被分割
        if parent in used_intervals:
            print(0)
            return
        used_intervals.add(parent)
        start, end = parent
        # 分割父区间，生成两个非空的连续子区间
        if a == start and b == end:
            # 不能将区间分割成自身
            print(0)
            return
        left = (start, a - 1) if a > start else None
        middle = (a, b)
        right = (b + 1, end) if b < end else None
        children = []
        if left and left[0] <= left[1]:
            children.append(left)
        if middle[0] <= middle[1]:
            children.append(middle)
        if right and right[0] <= right[1]:
            children.append(right)
        # 检查是否恰好分成两个非空子区间
        if len(children) != 2:
            print(0)
            return
        intervals[parent] = children
        for child in children:
            intervals[child] = []

    print(1)

if __name__ == "__main__":
    main()
