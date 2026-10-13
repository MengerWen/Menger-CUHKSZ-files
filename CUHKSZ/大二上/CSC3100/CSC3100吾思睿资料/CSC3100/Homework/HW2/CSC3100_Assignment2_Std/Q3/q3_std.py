k = int(input())
N = 3**k
pat = [['#']*N for _ in range(N)]

def write(s_row, s_col, level):
    if level == 0:
        return 
    L = 3 ** (level - 1)
    for i in range(s_row + L, s_row + 2 * L):
        for j in range(s_col + L, s_col + L + L):
            pat[i][j] = ' '
    write(s_row, s_col, level - 1)
    write(s_row + L, s_col, level - 1)
    write(s_row + 2*L, s_col, level - 1)
    write(s_row, s_col + L, level - 1)
    write(s_row + 2*L, s_col + L, level - 1)
    write(s_row, s_col + 2*L, level - 1)
    write(s_row + L, s_col + 2*L, level - 1)
    write(s_row + 2*L, s_col + 2*L, level - 1)

write(0, 0, k)
for s_pat in pat:
    print(''.join(s_pat))
