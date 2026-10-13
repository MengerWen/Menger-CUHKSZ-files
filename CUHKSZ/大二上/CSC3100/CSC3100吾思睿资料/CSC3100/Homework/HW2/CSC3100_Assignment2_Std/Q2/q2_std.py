ans = 0
def mergesort(a):
    global ans
    if len(a) == 1 :
        return a
    mid = len(a) // 2
    a1 = mergesort(a[:mid])
    a2 = mergesort(a[mid:])
    i = 0 
    j = 0
    b = []
    while i != len(a1) and j != len(a2):
        if(a1[i] < a2[j]):
            b.append(a1[i])
            i += 1
        else:
            ans += len(a1) - i
            b.append(a2[j])
            j += 1
    while i != len(a1):
        b.append(a1[i])
        i += 1
    while j != len(a2):
        b.append(a2[j])
        j += 1
    return b


if __name__ == '__main__':
    ans = 0
    n = int(input())
    A = input().split()
    for i in range(len(A)):
        A[i] = int(A[i])
    A = mergesort(A)
    print(ans)