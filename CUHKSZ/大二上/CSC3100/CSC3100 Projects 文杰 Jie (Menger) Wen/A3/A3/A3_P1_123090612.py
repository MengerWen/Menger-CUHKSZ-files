# Read the number of nodes, edges, and the root node t
n, m, t = map(int, input().split())

# Initialize adjacency list and degrees list
adj = [[] for _ in range(n + 1)]  # Using 1-based indexing
degrees = [0] * (n + 1)

# Read the edges and build the tree
for _ in range(m):
    u, v, w = map(int, input().split())
    adj[u].append((v, w))
    adj[v].append((u, w))
    degrees[u] += 1
    degrees[v] += 1

# Initialize necessary arrays
parent = [0] * (n + 1)
depth = [0] * (n + 1)
w_edge = [0] * (n + 1)
visited = [False] * (n + 1)

# Depth-First Search to compute parent, depth, and edge weights
def dfs(u, p):
    visited[u] = True
    for v, w in adj[u]:
        if not visited[v]:
            parent[v] = u
            depth[v] = depth[u] + 1
            w_edge[v] = w  # Weight of edge from v to parent[v]
            dfs(v, u)

dfs(t, 0)

# Collect all leaf nodes except the root node t
leaf_nodes = []
for u in range(1, n + 1):
    if degrees[u] == 1 and u != t:
        leaf_nodes.append(u)

max_total_HP = 0  # Initialize the maximum HP required

# Compute the total HP required for each leaf node
for u in leaf_nodes:
    total_HP = 0
    k = depth[u]  # Starting MP for this path
    curr = u
    while curr != t:
        w = w_edge[curr]  # Weight of the edge from curr to parent[curr]
        HP_consumed = max(0, w - k)
        total_HP += HP_consumed
        k -= 1  # Decrease MP by 1 as we move to the next node
        curr = parent[curr]
    max_total_HP = max(max_total_HP, total_HP)

# Output the minimum health point required at the beginning
print(max_total_HP)
