n, q = map(int, input().split())
permut = list(map(int, input().split()))
ls = list(map(int, input().split()))
rs = list(map(int, input().split()))

val_to_idx = [0] * (n + 1)

for idx, val in enumerate(permut):
    val_to_idx[val] = idx

ls = [val_to_idx[val] for val in ls]
rs = [val_to_idx[val] for val in rs]

is_valid = True

if ls[0] != 0 or rs[0] != n - 1:
    is_valid = False
else:
    l_pre, r_pre = ls[0], rs[0]
    for l, r in zip(ls[1:], rs[1:]):
        if (l == l_pre) :
            if (l < r < r_pre):
                r_pre = r
            else:
                is_valid = False
                break
        elif (r == r_pre):
            if (l_pre < l < r):
                l_pre = l
            else:
                is_valid = False
                break
        else:
            is_valid = False
            break

print(int(is_valid))