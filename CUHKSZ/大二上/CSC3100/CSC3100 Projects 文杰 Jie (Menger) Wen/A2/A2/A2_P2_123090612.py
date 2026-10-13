def will_loop_execute(x, y):
    if x == 'n' and y == 'n':
        return True
    elif x == 'n' and y != 'n':
        return False
    elif x != 'n' and y == 'n':
        return True
    else:
        x = int(x)
        y = int(y)
        return x <= y

t = int(input())
results = []
for _ in range(t):
    L = int(input())
    lines = [input().strip() for _ in range(L)]
    variables_in_use = set()
    exponent_stack = []
    idx = 0
    error = False
    program_exponents = []
    while idx < L:
        line = lines[idx]
        if line.startswith('F'):
            parts = line.split()
            if len(parts) != 4:
                error = True
                break
            _, i, x, y = parts

            if i in variables_in_use:
                # Variable already in use
                error = True
                break
            variables_in_use.add(i)

            loop_runs = will_loop_execute(x, y)

            if loop_runs:
                if x == 'n' and y == 'n':
                    loop_exponent = 0
                elif x == 'n' or y == 'n':
                    loop_exponent = 1
                else:
                    loop_exponent = 0
            else:
                loop_exponent = 0

            exponent_stack.append({'exponent': loop_exponent, 'child_exponents': [], 'variable': i})
        elif line == 'E':
            if not exponent_stack:
                # Unmatched 'E'
                error = True
                break
            scope = exponent_stack.pop()
            variables_in_use.remove(scope['variable'])
            total_exponent = scope['exponent']
            if scope['child_exponents']:
                total_exponent += max(scope['child_exponents'])
            # Pass total_exponent to parent scope or store it
            if exponent_stack:
                exponent_stack[-1]['child_exponents'].append(total_exponent)
            else:
                program_exponents.append(total_exponent)
        else:
            # Invalid line
            error = True
            break
        idx += 1

    if error or exponent_stack:
        results.append('ERR')
    else:
        if program_exponents:
            total_exponent = max(program_exponents)
        else:
            total_exponent = 0
        if total_exponent == 0:
            results.append('O(1)')
        else:
            results.append(f'O(n^{total_exponent})')

print("\n".join(results))
