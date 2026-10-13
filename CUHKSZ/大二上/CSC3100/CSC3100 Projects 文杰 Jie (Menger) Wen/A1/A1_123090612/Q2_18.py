def main():
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx
        
    splits = []  # 存储有效的分割区间
    split_set = set()  # 用于检查区间的唯一性
    invalid = False  # 标记输入是否无效
    
    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
    
        # 确保区间有效
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
    
    # 按照左端点升序、右端点降序排序
    splits.sort(key=lambda x: (x[0], -x[1]))
    
    if not splits:
        print(0)
        return
    
    first_split = splits[0]
    if first_split != (1, n):
        print(0)
        return
    
    # 使用栈和子区间信息来检查嵌套关系和完整覆盖
    stack = []
    children = {}  # 记录每个区间的子区间列表
    stack.append(first_split)
    children[first_split] = []
    
    for split in splits[1:]:
        a, b = split
        # 处理栈顶元素，找到当前区间的父区间
        while stack and not (a >= stack[-1][0] and b <= stack[-1][1]):
            parent = stack.pop()
            # 检查父区间的子区间是否满足条件
            if len(children[parent]) == 1:
                child_interval = children[parent][0]
                # 获取父区间的左、右端点
                p_left, p_right = parent
                # 获取子区间的左、右端点
                c_left, c_right = child_interval
                # 判断是否允许子区间与父区间的部分匹配
                if not (p_left <= c_left and c_right <= p_right):
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
        # 添加当前区间为父区间的子区间
        children[parent].append(split)
        stack.append(split)
        children[split] = []
    
    # 处理剩余的栈，检查最后的区间是否满足条件
    while stack:
        parent = stack.pop()
        if len(children[parent]) == 1:
            child_interval = children[parent][0]
            # 获取父区间的左、右端点
            p_left, p_right = parent
            # 获取子区间的左、右端点
            c_left, c_right = child_interval
            # 判断是否允许子区间与父区间的部分匹配
            if not (p_left <= c_left and c_right <= p_right):
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
        # 如果没有子区间，且不是整个排列，则无需处理
    print(1)

if __name__ == "__main__":
    main()
