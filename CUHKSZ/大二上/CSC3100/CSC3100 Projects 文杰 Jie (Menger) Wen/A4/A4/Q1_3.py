def main():
    import sys
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        def next_token():
            return next(it)
    except StopIteration:
        pass

    n = int(next_token())
    m = int(next_token())
    q = int(next_token())

    edges = [None]  # 1-based indexing
    adj = [[] for _ in range(n+1)]

    for edge_idx in range(1, m+1):
        u = int(next_token())
        v = int(next_token())
        w = int(next_token())
        edges.append( (u, v, w) )
        adj[u].append( (v, w, edge_idx) )
        adj[v].append( (u, w, edge_idx) )

    queries_required_edges = []
    for _ in range(q):
        k_i = int(next_token())
        required = []
        for _ in range(k_i):
            e_idx = int(next_token())
            required.append(e_idx)
        queries_required_edges.append(required)

    queries_s_t = []
    for _ in range(q):
        s_i = int(next_token())
        t_i = int(next_token())
        queries_s_t.append( (s_i, t_i) )

    INF = 10**18

    class MinHeap:
        def __init__(self):
            self.heap = []

        def push(self, item):
            self.heap.append(item)
            self._sift_up(len(self.heap) - 1)

        def pop(self):
            if not self.heap:
                return None
            item = self.heap[0]
            last_item = self.heap.pop()
            if self.heap:
                self.heap[0] = last_item
                self._sift_down(0)
            return item

        def _sift_up(self, idx):
            while idx > 0:
                parent = (idx - 1) // 2
                if self.heap[parent][0] > self.heap[idx][0]:
                    self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
                    idx = parent
                else:
                    break

        def _sift_down(self, idx):
            n = len(self.heap)
            while True:
                smallest = idx
                left = 2 * idx + 1
                right = 2 * idx + 2
                if left < n and self.heap[left][0] < self.heap[smallest][0]:
                    smallest = left
                if right < n and self.heap[right][0] < self.heap[smallest][0]:
                    smallest = right
                if smallest != idx:
                    self.heap[smallest], self.heap[idx] = self.heap[idx], self.heap[smallest]
                    idx = smallest
                else:
                    break

    for q_idx in range(q):
        required_edges = queries_required_edges[q_idx]
        s, t = queries_s_t[q_idx]
        k = len(required_edges)
        edge_to_bit = {}
        for i, e in enumerate(required_edges):
            edge_to_bit[e] = i
        target_mask = (1 << k) - 1
        dist = [ [INF] * (1 << k) for _ in range(n+1) ]
        dist[s][0] = 0
        heap = MinHeap()
        heap.push( (0, s, 0) )
        while heap.heap:
            current = heap.pop()
            d, u, mask = current
            if u == t and mask == target_mask:
                print(d)
                break
            if d > dist[u][mask]:
                continue
            for (v, w, e_idx) in adj[u]:
                new_mask = mask
                if e_idx in edge_to_bit:
                    bit = edge_to_bit[e_idx]
                    new_mask = mask | (1 << bit)
                if dist[v][new_mask] > d + w:
                    dist[v][new_mask] = d + w
                    heap.push( (d + w, v, new_mask) )
        else:
            print(-1)

if __name__ == "__main__":
    main()