# Priority Queue removed; using simple list operations instead

# Function to read a line and split into integers
def readints():
    while True:
        try:
            line = input()
            if line.strip() == '':
                continue
            return list(map(int, line.strip().split()))
        except EOFError:
            return []

def read_edge():
    while True:
        try:
            line = input()
            if line.strip() == '':
                continue
            return list(map(int, line.strip().split()))
        except EOFError:
            return []

def main():
    # Read n, m, q
    first_line = readints()
    while not first_line:
        first_line = readints()
    n, m, q = first_line

    # Read m edges
    edges = [None]  # 1-based indexing
    graph = [[] for _ in range(n + 1)]
    for edge_idx in range(1, m + 1):
        edge = read_edge()
        while not edge:
            edge = read_edge()
        u, v, w = edge
        edges.append((u, v, w))
        graph[u].append((v, w, edge_idx))
        graph[v].append((u, w, edge_idx))

    # Read q queries' E_j lists
    queries_E = []
    for _ in range(q):
        # Read k_i
        ki_list = readints()
        while not ki_list:
            ki_list = readints()
        k_i = ki_list[0]
        # Read k_i edge indices
        E_j = []
        while len(E_j) < k_i:
            e_list = readints()
            if not e_list:
                continue
            E_j += e_list
        queries_E.append(E_j[:k_i])

    # Read q lines of s_j and t_j
    queries_ST = []
    for _ in range(q):
        st_list = readints()
        while not st_list:
            st_list = readints()
        s_j, t_j = st_list
        queries_ST.append((s_j, t_j))

    # Function to push to priority queue (list)
    def push_pq(pq, item):
        pq.append(item)

    # Function to pop from priority queue (list)
    def pop_pq(pq):
        if not pq:
            return None
        # Find the item with the smallest first element
        min_idx = 0
        min_item = pq[0]
        for idx in range(1, len(pq)):
            if pq[idx][0] < min_item[0]:
                min_item = pq[idx]
                min_idx = idx
        # Remove and return the smallest item
        return pq.pop(min_idx)

    # Process each query
    for query_idx in range(q):
        E_j = queries_E[query_idx]
        s_j, t_j = queries_ST[query_idx]
        k_i = len(E_j)
        if k_i == 0:
            # If there are no edges to traverse, simply find the shortest path
            # using standard Dijkstra's algorithm without mask
            # Initialize priority queue as a list
            pq = []
            push_pq(pq, (0, s_j))
            # Initialize visited list
            visited = [False] * (n + 1)
            answer_found = False
            while pq:
                current = pop_pq(pq)
                if current is None:
                    break
                current_time, current_node = current
                if current_node == t_j:
                    print(current_time)
                    answer_found = True
                    break
                if visited[current_node]:
                    continue
                visited[current_node] = True
                for neighbor in graph[current_node]:
                    v, w_edge, _ = neighbor
                    if not visited[v]:
                        push_pq(pq, (current_time + w_edge, v))
            if not answer_found:
                print(-1)
            continue

        # Create a mapping from edge_index to bit position
        E_j_map = {}
        for bit_pos in range(k_i):
            edge_index = E_j[bit_pos]
            E_j_map[edge_index] = bit_pos
        target_mask = (1 << k_i) - 1

        # Initialize Dijkstra
        pq = []
        push_pq(pq, (0, s_j, 0))
        # Initialize visited list as a list of sets
        visited = [set() for _ in range(n + 1)]
        answer_found = False

        while pq:
            current = pop_pq(pq)
            if current is None:
                break
            current_time, current_node, mask = current
            # If reached destination with all edges traversed
            if current_node == t_j and mask == target_mask:
                print(current_time)
                answer_found = True
                break
            # If this mask is already visited for current_node
            if mask in visited[current_node]:
                continue
            # Mark this mask as visited for current_node
            visited[current_node].add(mask)
            # Traverse neighbors
            for neighbor in graph[current_node]:
                v, w_edge, edge_idx = neighbor
                new_mask = mask
                if edge_idx in E_j_map:
                    bit = E_j_map[edge_idx]
                    new_mask = mask | (1 << bit)
                push_pq(pq, (current_time + w_edge, v, new_mask))
        if not answer_found:
            # If no path found, print -1
            print(-1)

if __name__ == "__main__":
    main()