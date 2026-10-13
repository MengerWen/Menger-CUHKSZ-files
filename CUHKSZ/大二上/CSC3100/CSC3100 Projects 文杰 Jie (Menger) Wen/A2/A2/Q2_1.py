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
for _ in range(t):
    L = int(input())
    lines = [input().strip() for _ in range(L)]
    variables_in_use = set()
    exponent_stack = []
    sequential_blocks = []
    active_stack = []
    idx = 0

    while idx < L:
        line = lines[idx]
        if line.startswith('F'):
            parts = line.split()
            i = parts[1]
            x = parts[2]
            y = parts[3]

            if i in variables_in_use:
                pass

            current_active = True if not active_stack or active_stack[-1] == 'active' else False
            loop_runs = will_loop_execute(x, y)

            if current_active and loop_runs:
                active_stack.append('active')
                variables_in_use.add(i)
                if y == 'n' and x != 'n':
                    loop_exponent = 1
                elif y == 'n' and x == 'n':
                    loop_exponent = 0
                else:
                    loop_exponent = 0
                exponent_stack.append([loop_exponent, i])
            else:
                active_stack.append('inactive')
                exponent_stack.append([0, i])
        elif line == 'E':
            status = active_stack.pop()
            loop_exponent, i = exponent_stack.pop()
            if status == 'active':
                variables_in_use.remove(i)
                if exponent_stack:
                    if active_stack[-1] == 'active':
                        exponent_stack[-1][0] += loop_exponent
                else:
                    sequential_blocks.append(loop_exponent)
        idx += 1

    if sequential_blocks:
        total_exponent = max(sequential_blocks)
    else:
        total_exponent = 0

    if total_exponent == 0:
        print('O(1)')
    else:
        print(f'O(n^{total_exponent})')