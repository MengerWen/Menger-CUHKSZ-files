import sys

def main():
    import sys
    import threading

    def run():
        import sys

        data = sys.stdin.read().split()
        idx = 0
        n = int(data[idx]); idx +=1
        m = int(data[idx]); idx +=1
        P = int(data[idx]); idx +=1

        a = [0]*(n+1)
        counts = [0]*(P+1)

        sum_total =0

        for i in range(1,n+1):
            val = int(data[idx]); idx +=1
            a[i]=val
            abs_val = abs(val)
            if abs_val <= P:
                counts[abs_val] +=1
            sum_total +=val

        # Initialize total_distinct
        total_distinct = 0
        if P >=0 and counts[0]>=1:
            total_distinct +=1
        for x in range(1,P+1):
            if counts[x] >=1:
                total_distinct +=1
            if counts[x] >=2:
                total_distinct +=1

        output = []
        for _ in range(m):
            if idx >= len(data):
                break
            op = data[idx]; idx +=1
            if op == '1':
                if idx +4 > len(data):
                    break
                k = int(data[idx]); idx +=1
                x = int(data[idx]); idx +=1
                y = int(data[idx]); idx +=1
                c = int(data[idx]); idx +=1

                old_val = a[k]
                old_abs = abs(old_val)
                # Update total_distinct for old_abs
                if old_abs <= P:
                    if old_abs ==0:
                        if counts[old_abs] >=1:
                            if counts[old_abs] ==1:
                                total_distinct -=1
                            elif counts[old_abs] ==2:
                                pass
                            elif counts[old_abs] >2:
                                pass
                    else:
                        if counts[old_abs] >=2:
                            total_distinct -=1
                        if counts[old_abs] >=1:
                            total_distinct -=1
                    counts[old_abs] -=1
                    # Recalculate total_distinct
                    if old_abs ==0:
                        if counts[old_abs] >=1:
                            total_distinct +=1
                    else:
                        if counts[old_abs] >=1:
                            total_distinct +=1
                        if counts[old_abs] >=2:
                            total_distinct +=1

                # Compute new_val
                new_val = ((x*x + k*y +5*x) % P) *c
                a[k] = new_val
                sum_total += new_val - old_val
                new_abs = abs(new_val)
                if new_abs <= P:
                    if new_abs ==0:
                        if counts[new_abs] >=1:
                            if counts[new_abs] ==1:
                                total_distinct +=1
                            elif counts[new_abs] ==2:
                                pass
                            elif counts[new_abs] >2:
                                pass
                    else:
                        if counts[new_abs] >=1:
                            total_distinct +=1
                        if counts[new_abs] >=2:
                            total_distinct +=1
                    counts[new_abs] +=1
                    # Recalculate total_distinct
                    if new_abs ==0:
                        if counts[new_abs] ==1:
                            total_distinct +=1
                    else:
                        if counts[new_abs] ==1:
                            total_distinct +=1
                        if counts[new_abs] ==2:
                            total_distinct +=1

            elif op == '2':
                output.append(str(sum_total))
            elif op == '3':
                output.append(str(total_distinct))
        print('\n'.join(output))

    threading.Thread(target=run).start()
