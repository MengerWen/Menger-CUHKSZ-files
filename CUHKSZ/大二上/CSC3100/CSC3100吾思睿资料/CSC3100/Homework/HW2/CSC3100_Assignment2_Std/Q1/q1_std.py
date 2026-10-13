def represent(x):
    p = 0
    binary = []
    ans = []
    while x:
        if x % 2 == 1:
            binary.append(p)
        x = x // 2
        p += 1
    
    while binary:
        p = binary[-1]
        binary.pop(-1)
        if p == 0:
            ans.append("2(0)")
        elif p == 1:
            ans.append("2")
        else:
            ans += ["2(", *represent(p), ")"]
        if binary:
            ans.append('+')

    return ans
    

n = int(input())
print(''.join(represent(n)))