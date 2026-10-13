def main():
    # Read the number of flowerbeds and paths
    n, m = map(int, input().split())
    
    # Read the edges and store them in a list
    edge_list = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        if u > v:
            u, v = v, u
        edge_list.append([u, v, w])
    
    # Sort the edge list based on the flowerbeds to facilitate binary search
    edge_list.sort(key=lambda x: (x[0], x[1]))
    
    # Build the adjacency list
    adj = [[] for _ in range(n + 1)]
    for idx in range(len(edge_list)):
        u, v, w = edge_list[idx]
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Read the number of changes
    q = int(input())
    changes = []
    
    # Read all the changes
    for _ in range(q):
        k_i = int(input())
        change_block = []
        for __ in range(k_i):
            a, b, c = map(int, input().split())
            change_block.append((a, b, c))
        changes.append(change_block)
    
    # Read all the queries (q + 1)
    queries = []
    for _ in range(q + 1):
        s_i, t_i = map(int, input().split())
        queries.append((s_i, t_i))
    
    # Function to find the index of an edge in the sorted edge_list
    def find_edge_index(a, b):
        if a > b:
            a, b = b, a
        l = 0
        r = len(edge_list) - 1
        while l <= r:
            mid = (l + r) // 2
            if edge_list[mid][0] == a and edge_list[mid][1] == b:
                return mid
            elif edge_list[mid][0] < a or (edge_list[mid][0] == a and edge_list[mid][1] < b):
                l = mid + 1
            else:
                r = mid - 1
        return -1  # Edge not found
    
    # Function to check if there's a path from s to t with all edges having at least w flowers
    def is_connected(s, t, w):
        if s == t:
            return True
        visited = [False] * (n + 1)
        queue = [s]
        visited[s] = True
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for (v, edge_idx) in adj[u]:
                if not visited[v] and edge_list[edge_idx][2] >= w:
                    if v == t:
                        return True
                    visited[v] = True
                    queue.append(v)
        return False
    
    # Function to find the maximum bottleneck value between s and t using binary search
    def find_max_bottleneck(s, t):
        low = 0
        high = max(edge[2] for edge in edge_list) if edge_list else 0
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if is_connected(s, t, mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans
    
    # Process each day and handle the changes
    for day in range(q + 1):
        if day > 0:
            # Apply the (day-1)th change
            change = changes[day - 1]
            for (a, b, c) in change:
                idx = find_edge_index(a, b)
                if idx != -1:
                    edge_list[idx][2] = c
        # Handle the query for the current day
        s, t = queries[day]
        res = find_max_bottleneck(s, t)
        print(res)

# Execute the main function
if __name__ == "__main__":
    main()