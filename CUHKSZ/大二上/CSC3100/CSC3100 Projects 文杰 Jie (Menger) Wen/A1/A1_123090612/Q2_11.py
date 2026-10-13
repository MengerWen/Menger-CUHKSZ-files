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
        # 从栈中弹出，找到当前区间的父区间
        while stack and not (a >= stack[-1][0] and b <= stack[-1][1]):
            parent = stack.pop()
            # 处理父区间，检查子区间覆盖情况
            child_intervals = children[parent]
            # 添加隐式子区间
            intervals = child_intervals.copy()
            prev_end = parent[0] - 1
            total_intervals = []
            for interval in sorted(intervals, key=lambda x: x[0]):
                if interval[0] > prev_end + 1:
                    # 存在间隙，添加隐式区间
                    total_intervals.append((prev_end + 1, interval[0] - 1))
                total_intervals.append(interval)
                prev_end = interval[1]
            if prev_end < parent[1]:
                # 添加末尾的隐式区间
                total_intervals.append((prev_end + 1, parent[1]))
            # 检查总的子区间数量是否为2
            if len(total_intervals) != 2:
                print(0)
                return
            # 检查子区间是否完整覆盖父区间且没有重叠
            coverage = 0
            for i, interval in enumerate(total_intervals):
                if i > 0 and total_intervals[i-1][1] + 1 != interval[0]:
                    print(0)
                    return
                coverage += interval[1] - interval[0] + 1
            if coverage != parent[1] - parent[0] + 1:
                print(0)
                return
        if not stack:
            print(0)
            return
        parent = stack[-1]
        if split[0] < parent[0] or split[1] > parent[1]:
            print(0)
            return
        children.setdefault(parent, []).append(split)
        stack.append(split)
        children[split] = []
    
    # 处理剩余的栈，检查最后的区间是否满足条件
    while stack:
        parent = stack.pop()
        child_intervals = children[parent]
        # 添加隐式子区间
        intervals = child_intervals.copy()
        prev_end = parent[0] - 1
        total_intervals = []
        for interval in sorted(intervals, key=lambda x: x[0]):
            if interval[0] > prev_end + 1:
                # 存在间隙，添加隐式区间
                total_intervals.append((prev_end + 1, interval[0] - 1))
            total_intervals.append(interval)
            prev_end = interval[1]
        if prev_end < parent[1]:
            # 添加末尾的隐式区间
            total_intervals.append((prev_end + 1, parent[1]))
        # 检查总的子区间数量是否为2
        if len(total_intervals) != 2:
            print(0)
            return
        # 检查子区间是否完整覆盖父区间且没有重叠
        coverage = 0
        for i, interval in enumerate(total_intervals):
            if i > 0 and total_intervals[i-1][1] + 1 != interval[0]:
                print(0)
                return
            coverage += interval[1] - interval[0] + 1
        if coverage != parent[1] - parent[0] + 1:
            print(0)
            return
    print(1)

if __name__ == "__main__":
    main()
