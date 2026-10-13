def main():
    n, q = map(int, input().split())
    permutation = list(map(int, input().split()))
    l_list = list(map(int, input().split()))
    r_list = list(map(int, input().split()))

    position = {}
    for idx, val in enumerate(permutation):
        position[val] = idx

    parent = {}
    min_pos = {}
    max_pos = {}
    size = {}

    # Initialize Union-Find structures
    for val in permutation:
        parent[val] = val
        pos = position[val]
        min_pos[val] = pos
        max_pos[val] = pos
        size[val] = 1

    def find(u):
        # Iterative find with path compression
        root = u
        while parent[root] != root:
            root = parent[root]
        while parent[u] != u:
            parent[u], u = root, parent[u]
        return root

    valid = True

    # Process operations in reverse
    for i in range(q - 1, -1, -1):
        x = l_list[i]
        y = r_list[i]

        if x not in parent or y not in parent:
            valid = False
            break

        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            new_min = min(min_pos[root_x], min_pos[root_y])
            new_max = max(max_pos[root_x], max_pos[root_y])
            total_size = new_max - new_min + 1
            total_elements = size[root_x] + size[root_y]
            if total_size != total_elements:
                valid = False
                break
            else:
                parent[root_x] = root_y
                min_pos[root_y] = new_min
                max_pos[root_y] = new_max
                size[root_y] = total_elements

    print(1 if valid else 0)

if __name__ == '__main__':
    main()
