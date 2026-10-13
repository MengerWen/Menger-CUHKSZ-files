def main():
    n, m, P = map(int, input().split())
    a_list = input().split()

    a = []
    total_sum = 0
    for val_str in a_list:
        val = int(val_str)
        a.append(val)
        total_sum += val

    count_zero = 0
    counts = {}

    for val in a:
        abs_val = abs(val)
        if abs_val == 0:
            count_zero += 1
        else:
            counts[abs_val] = counts.get(abs_val, 0) + 1

    for _ in range(m):
        ops = input().split()
        if not ops:
            continue

        # Update operations
        if ops[0] == '1':
            _, k_str, x_str, y_str, c_str = ops
            k = int(k_str) - 1  # zero-based index
            x = int(x_str)
            y = int(y_str)
            c = int(c_str)

            old_val = a[k]
            old_abs = abs(old_val)
            # Update counts for old value
            if old_abs == 0:
                count_zero -= 1
            else:
                counts[old_abs] -= 1
                if counts[old_abs] == 0:
                    del counts[old_abs]

            # Compute new value
            new_val = ((x * x + (k + 1) * y + 5 * x) % P) * c
            a[k] = new_val
            new_abs = abs(new_val)

            # Update counts for new value
            if new_abs == 0:
                count_zero += 1
            else:
                counts[new_abs] = counts.get(new_abs, 0) + 1

            # Update total sum
            total_sum += new_val - old_val

        # Sum queries
        elif ops[0] == '2':
            print(total_sum)
        
        # Distinct value queries
        elif ops[0] == '3':
            max_distinct = 0
            if count_zero > 0:
                max_distinct += 1  # Zero contributes at most 1
            for count_v in counts.values():
                max_distinct += min(count_v, 2)
            print(max_distinct)
        
        else:
            pass  # Should not occur based on problem description

main()
