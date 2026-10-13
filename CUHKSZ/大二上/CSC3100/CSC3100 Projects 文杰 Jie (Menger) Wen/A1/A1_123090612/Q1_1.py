import sys

def main():
    import sys    
    def input():
        return sys.stdin.read()
    
    data = sys.stdin.read().split()
    ptr = 0
    n, m, P = int(data[ptr]), int(data[ptr+1]), int(data[ptr+2])
    ptr +=3
    a = []
    counts = [0]*(P+1)
    for i in range(n):
        val = int(data[ptr])
        ptr +=1
        a.append(val)
        abs_val = abs(val)
        if abs_val <= P:
            counts[abs_val] +=1
    # Initialize unique_abs_count
    unique_abs_count = 0
    for x in range(P+1):
        if counts[x] >=1:
            unique_abs_count +=1
    # Initialize total_sum
    total_sum = sum(a)
    # Prepare output
    output = []
    for _ in range(m):
        op = data[ptr]
        ptr +=1
        if op == '1':
            k = int(data[ptr]) -1
            x = int(data[ptr+1])
            y = int(data[ptr+2])
            c = int(data[ptr+3])
            ptr +=4
            old_val = a[k]
            old_abs = abs(old_val)
            counts[old_abs] -=1
            if counts[old_abs] ==0:
                unique_abs_count -=1
            # Compute new_val
            new_val = ((x*x + (k+1)*y +5*x) % P) * c
            a[k] = new_val
            new_abs = abs(new_val)
            if new_abs <= P:
                if counts[new_abs] ==0:
                    unique_abs_count +=1
                counts[new_abs] +=1
            # Update total_sum
            total_sum += new_val - old_val
        elif op == '2':
            output.append(str(total_sum))
        elif op == '3':
            output.append(str(unique_abs_count))
    print('\n'.join(output))

if __name__ == "__main__":
    main()
