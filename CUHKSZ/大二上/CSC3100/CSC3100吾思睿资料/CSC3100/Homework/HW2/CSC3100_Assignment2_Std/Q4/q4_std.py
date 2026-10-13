class node:
    def __init__(self, val):
        self.sz = 1
        self.val = val
        self.sum = val
        self.l = None
        self.r = None

root = None

def up(tmp):
    tmp.sz = 1
    tmp.sum = tmp.val
    if tmp.l != None :
        tmp.sz += tmp.l.sz
        tmp.sum += tmp.l.sum
    if tmp.r != None :
        tmp.sz += tmp.r.sz
        tmp.sum += tmp.r.sum

def insert(tmp, k, x):
    if tmp == None:
        return node(x)
    if tmp.l != None :
        lsz = tmp.l.sz
    else:
        lsz = 0
    if k <= lsz:
        tmp.l = insert(tmp.l, k, x)
    else:
        tmp.r = insert(tmp.r, k - lsz - 1, x)
    up(tmp)
    return tmp

def delpre(tmp):
    if tmp.r == None:
        return tmp
    if tmp.r.r == None :
        ans = delpre(tmp.r)
        tmp.r = tmp.r.l
        up(tmp)
        return ans
    ans = delpre(tmp.r)
    up(tmp)
    return ans

def delete(tmp, k):
    if tmp.l != None :
        lsz = tmp.l.sz
    else:
        lsz = 0
    if k == lsz + 1 :
        if tmp.l == None:
            return tmp.r
        elif tmp.r == None:
            return tmp.l
        elif tmp.l.r == None:
            tmp.val = tmp.l.val
            tmp.l = tmp.l.l
            up(tmp)
            return tmp
        newval = delpre(tmp.l).val
        tmp.val = newval
        up(tmp)
        return tmp
    if(k <= lsz):
        tmp.l = delete(tmp.l, k)
    else:
        tmp.r = delete(tmp.r, k - lsz - 1)
    up(tmp)
    return tmp

def getsum(tmp,k):
    if(tmp == None):
        return 0
    if tmp.l != None :
        lsz = tmp.l.sz
    else:
        lsz = 0
    if k <= lsz:
        return getsum(tmp.l,k)
    if(tmp.l != None):
        ans = tmp.l.sum
    else:
        ans = 0
    if k == lsz + 1:
        return ans + tmp.val
    else:
        return ans + tmp.val + getsum(tmp.r, k - lsz - 1)

if __name__ == '__main__':
    root = None
    n = int(input())
    for t in range(n):
        s = input().split()
        for i in range(len(s)):
            s[i] = int(s[i])
        if(s[0] == 1):
            root = insert(root,s[1],s[2])
        if(s[0] == 2):
            root = delete(root,s[1])
        if(s[0] == 3):
            print(getsum(root,s[2])-getsum(root,s[1]-1))