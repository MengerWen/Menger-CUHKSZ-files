import sys

def main():
    import sys
    from sys import stdin

    import sys

    tokens = sys.stdin.read().split()
    it = iter(tokens)

    n = int(next(it))
    m = int(next(it))
    P = int(next(it))

    a = [int(next(it)) for _ in range(n)]

    counts = [0]*(P+1)
    for num in a:
        v = abs(num)
        counts[v] +=1

    # Initialize num_v_eq1 and num_v_ge2
    num_v_eq1 = counts[1:].count(1)
    num_v_ge2 = sum(1 for c in counts[1:] if c >=2)
    num_zero_present = counts[0] >=1

    sum_a = sum(a)

    outputs = []
    for _ in range(m):
        try:
            op = next(it)
        except StopIteration:
            break
        if op == '1':
            k = int(next(it)) -1
            x = int(next(it))
            y = int(next(it))
            c = int(next(it))
            # Compute new_a_k
            new_val = ((x**2 + (k+1)*y + 5*x) % P) * c
            old_val = a[k]
            # Update sum_a
            sum_a = sum_a - old_val + new_val
            # Update counts and variables
            old_v = abs(old_val)
            new_v = abs(new_val)
            if old_v != new_v:
                # Handle old_v
                if old_v >0:
                    if counts[old_v] ==1:
                        num_v_eq1 -=1
                    elif counts[old_v] ==2:
                        num_v_ge2 -=1
                        num_v_eq1 +=1
                else:
                    if counts[0] ==1:
                        num_zero_present = False
                counts[old_v] -=1

                # Handle new_v
                counts[new_v] +=1
                if new_v >0:
                    if counts[new_v] ==1:
                        num_v_eq1 +=1
                    elif counts[new_v] ==2:
                        num_v_eq1 -=1
                        num_v_ge2 +=1
                else:
                    if counts[0] ==1:
                        num_zero_present = True
            # Update a[k]
            a[k] = new_val
        elif op == '2':
            outputs.append(str(sum_a))
        elif op == '3':
            max_distinct = num_v_eq1 + 2*num_v_ge2 + (1 if num_zero_present else0)
            outputs.append(str(max_distinct))
    print('\n'.join(outputs))

if __name__ == '__main__':
    main()
