def check_interval(l_pre, r_pre, l, r) -> tuple[int, int]:
    if (l == l_pre) :
        if (l < r <= r_pre):
            return (r + 1, r_pre)
    elif (r == r_pre):
        if (l_pre <= l < r):
            return (l_pre, l - 1)
    return None


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
    leaf_intervals= set([(ls[0], rs[0])])
    for l, r in zip(ls[1:], rs[1:]):
        is_father_found = False
        for (l_pre, r_pre) in leaf_intervals:
            # Find the possible father interval
            if (l_pre <= l <= r <= r_pre):
                is_father_found = True
                break

        if not is_father_found:
            is_valid = False
            break

        leaf_intervals.remove((l_pre, r_pre))
        another_leaf = check_interval(l_pre, r_pre, l, r)
        if another_leaf is not None:
            leaf_intervals.add((l, r))
            leaf_intervals.add(another_leaf)
        else:
            is_valid = False
            break

print(int(is_valid))