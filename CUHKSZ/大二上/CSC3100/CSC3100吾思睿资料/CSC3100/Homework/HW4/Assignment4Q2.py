import heapq
def query_median():
    for i in range(2*K):
        print(a[i], end = " ")
    print()

def adjust_heaps():
    global n, n1, n2
    global a
    global Q1
    global Q2
    t = (n+1)//2 - K
    while n1 > t:
        w = -heapq.heappop(Q1)
        heapq.heappush(Q2, a[-1])
        for i in range(2*K-1, 0, -1):
            a[i] = a[i-1]
        a[0] = w
        n1 -= 1
        n2 += 1
    
    while n1 < t:
        w = heapq.heappop(Q2)
        heapq.heappush(Q1, -a[0])
        for i in range(1, 2*K):
            a[i-1] = a[i]
        a[-1] = w
        n1 += 1
        n2 -= 1

def ins(w):
    global n, n1, n2
    n += 1
    median = a[K]
    if w < a[0]:
        heapq.heappush(Q1, -w)
        n1 += 1
    else :
        for i in range(2*K):
            if a[i] > w :
                a[i], w = w, a[i]
        heapq.heappush(Q2, w)
        n2 += 1
    adjust_heaps()

def del_median(p):
    global n, n1, n2
    n -= 1
    n1 -= 1
    global Q1
    w = -heapq.heappop(Q1)
    for i in range(p-1, 0, -1):
        a[i] = a[i-1]
    a[0] = w
    adjust_heaps()

m, K = input().split()
m = eval(m)
K = eval(K)
a = []
Q1 = []
Q2 = []
inp = input().split()
for i in range(2*K):
    a.append(eval(inp[i]))
a.sort()
n, n1, n2 = 2*K, 0, 0
for i in range(m):
    inp = input().split()
    o = eval(inp[0])
    if o == 1:
        ins(eval(inp[1]))
    if o == 2:
        query_median()
    if o == 3:
        del_median(eval(inp[1]))
