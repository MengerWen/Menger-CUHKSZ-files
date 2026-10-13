def main():
    # 读取输入
    n, q = map(int, input().split())
    P = list(map(int, input().split()))
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    
    # pos数组存储P中每个元素的索引
    pos = [0] * (n + 1)
    for idx, val in enumerate(P, start=1):
        pos[val] = idx

    splits = []
    split_set = set()

    for i in range(q):
        a = pos[l[i]]
        b = pos[r[i]]
        if a > b:
            a, b = b, a  # 保证a <= b
        key = (a, b)
        if key in split_set:
            # 如果相同的段重复，直接返回无效
            print(0)
            return
        split_set.add(key)
        splits.append(key)

    # 按照区间的起点进行排序，如果起点相同则按终点排序
    splits.sort(key=lambda x: (x[0], x[1]))

    # 检查是否这些区间能构成完整的覆盖
    current_end = 0
    for a, b in splits:
        if a > current_end + 1:
            # 出现了空隙，无法连续覆盖
            print(0)
            return
        current_end = max(current_end, b)

    # 检查最终是否覆盖了整个数组
    if current_end == n:
        print(1)
    else:
        print(0)

if __name__ == "__main__":
    main()
