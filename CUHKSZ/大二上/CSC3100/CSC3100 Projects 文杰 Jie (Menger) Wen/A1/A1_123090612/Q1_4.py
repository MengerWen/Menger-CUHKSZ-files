def main():
    def read_input():
        # Read all input and split by whitespace
        return input().split()
    
    # Reading data from input
    data = read_input()
    ptr = 0
    
    # Parsing first set of input values
    n, m, P = int(data[ptr]), int(data[ptr + 1]), int(data[ptr + 2])
    ptr += 3
    
    # Initialize list a and counts array
    a = []
    counts = [0] * (P + 1)
    
    # Reading the array a and counting absolute values
    for i in range(n):
        val = int(data[ptr])
        ptr += 1
        a.append(val)
        abs_val = abs(val)
        if abs_val <= P:
            counts[abs_val] += 1

    # Initialize unique_abs_count
    unique_abs_count = sum(1 for x in range(P + 1) if counts[x] >= 1)
    
    # Initialize total_sum
    total_sum = sum(a)
    
    # Prepare output
    output = []
    
    # Process each operation
    for _ in range(m):
        op = data[ptr]
        ptr += 1
        if op == '1':
            # Parsing the arguments of operation '1'
            k = int(data[ptr]) - 1
            x = int(data[ptr + 1])
            y = int(data[ptr + 2])
            c = int(data[ptr + 3])
            ptr += 4
            
            # Update old value and adjust unique_abs_count
            old_val = a[k]
            old_abs = abs(old_val)
            counts[old_abs] -= 1
            if counts[old_abs] == 0:
                unique_abs_count -= 1
            
            # Compute new_val
            new_val = ((x * x + (k + 1) * y + 5 * x) % P) * c
            a[k] = new_val
            new_abs = abs(new_val)
            
            # Update counts for new value and adjust unique_abs_count
            if new_abs <= P:
                if counts[new_abs] == 0:
                    unique_abs_count += 1
                counts[new_abs] += 1
            
            # Update total_sum
            total_sum += new_val - old_val
        
        elif op == '2':
            # Output the current total_sum
            output.append(str(total_sum))
        
        elif op == '3':
            # Output the current unique_abs_count
            output.append(str(unique_abs_count))
    
    # Print all the output results at once
    print('\n'.join(output))

if __name__ == "__main__":
    main()
