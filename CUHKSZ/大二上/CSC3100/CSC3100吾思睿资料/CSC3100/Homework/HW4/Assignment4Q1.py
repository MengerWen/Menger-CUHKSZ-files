import heapq

'''
class Heap:
    def __init__(self):
        self.n = 0
        self.a = [(1e9, 0)]

    def up(self, p):
        if p == 1:
            return
        q = p // 2
        if self.a[q] < self.a[p] :
            self.a[q], self.a[p] = self.a[p], self.a[q]
            self.up(q)

    def down(self, p):
        q = p * 2
        r = p * 2 + 1
        if q > self.n:
            self.a[p] = self.a[self.n]
            return
        if r <= self.n and self.a[r] > self.a[q]:
            self.a[p] = self.a[r]
            self.down(r)
        else :
            self.a[p] = self.a[q]
            self.down(q)

    def push(self, w):
        self.n += 1
        if len(self.a) == self.n :
            self.a.append(w)
        else :
            self.a[self.n] = w
        self.up(self.n)

    def empty(self):
        return (self.n == 0)
    
    def top(self):
        return self.a[1]
    
    def pop(self):
        w = self.a[1]
        self.down(1)
        self.n -= 1
        return w
'''

n, m, K = input().split()
n = eval(n)
m = eval(m)
K = eval(K)
inf = 1000000000
mp = [{} for i in range(0, n+1)]
G = [[] for i in range(0, n+1)]
for i in range(0, m) : 
    u, v, w = input().split()
    u = eval(u)
    v = eval(v)
    w = eval(w)
    G[u].append((w, v))
    G[v].append((w, u))

tt = n
for u in range(1, n+1):
    G[u].sort()
    for i, e in enumerate(G[u]):
        if i == 0 or (e[0] != G[u][i-1][0]) :
            tt += 1
            mp[u][e[0]] = tt

g = [[] for i in range(0, tt+1)]

for u in range(1, n+1):
    for i, e in enumerate(G[u]):
        w, v = e[0], e[1]
        uw = mp[u][w]
        g[u].append((uw, w))
        g[uw].append((v, w))
        if mp[v].get(K*w, -1) != -1 :
            vw = mp[v][K*w]
            g[uw].append((vw, (K-1)*w))

vis = [0 for i in range(0, tt+1)]
dis = [inf for i in range(0, tt+1)]

dis[1] = 0
#Q = Heap()
#Q.push((0, 1))
Q = []
heapq.heappush(Q, (0, 1))
#while not Q.empty() :
while len(Q):
    t = heapq.heappop(Q)
    x = t[1]
#    t = Q.top()
#    x = t[1]
#    Q.pop()
    if vis[x] : continue
    vis[x] = 1
    for i, e in enumerate(g[x]):
        y, w = e[0], e[1]
        if dis[y] > dis[x] + w:
            dis[y] = dis[x] + w 
            heapq.heappush(Q, (dis[y], y))
#            Q.push((-dis[y], y))

for u in range(1, n+1):
    d  = -1 if dis[u] == inf else dis[u]//2
    print(d, end = " ")
print()