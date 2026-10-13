def main():
    def solve():
        n, q = map(int, input().split())
        P = list(map(int, input().split()))
        l = list(map(int, input().split()))
        r = list(map(int, input().split()))
        
        pos = [0] * (n + 1)
        for idx, val in enumerate(P, start=1):
            pos[val] = idx
        
        intervals = []
        for i in range(q):
            l_i = l[i]
            r_i = r[i]
            pos_l = pos[l_i]
            pos_r = pos[r_i]
            L_i = min(pos_l, pos_r)
            R_i = max(pos_l, pos_r)
            intervals.append((L_i, R_i))
        
        # Check if the first interval covers the whole permutation
        intervals.sort()
        if intervals[0][0] != 1 or intervals[0][1] != n:
            print(0)
            return
        
        prev_L, prev_R = intervals[0]
        for i in range(1, q):
            L_i, R_i = intervals[i]
            if L_i > prev_L:
                if R_i <= prev_R:
                    prev_L, prev_R = L_i, R_i
                else:
                    print(0)
                    return
            elif L_i == prev_L:
                if R_i < prev_R:
                    prev_L, prev_R = L_i, R_i
                else:
                    print(0)
                    return
            else:
                print(0)
                return
        print(1)
    
    solve()

if __name__ == "__main__":
    main()
