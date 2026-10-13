def solve_supermarket_problem(n, k, bag_size, items):
    # Step 1: Organize items by shelf
    shelves = [[] for i in range(k)]
    for item_id, value in items:
        shelf_num = item_id % k
        shelves[shelf_num].append((item_id, value))
    
    # Sort items on each shelf by ID in descending order
    for shelf in shelves:
        shelf.sort(key=lambda x: x[0], reverse=True)
        print(shelf) #tentatively
    
    # Step 2: Create a linear representation considering the ring structure
    linear_shelves = shelves + shelves  # Double the array to handle circular nature
    #print(linear_shelves) #tentatively

    max_value = 0
    # Try all possible starting positions
    for start in range(k):
        # Try all possible lengths up to bag_size
        for length in range(1, bag_size + 1):
            # Try all valid subsequences
            for i in range(start, start + k):
                # Get continuous sequence of shelves
                current_sequence = linear_shelves[i:i + length]
                
                # Count items from each shelf
                shelf_counts = {}
                total_items = 0
                total_value = 0
                
                # Calculate items and values for current sequence
                #print('enumerate current_sequence',list(enumerate(current_sequence))) #tentatively
                for shelf_idx, shelf in enumerate(current_sequence):
                    if shelf:  # If shelf is not empty
                        count = min(bag_size - total_items, len(shelf))
                        if count > 0:
                            shelf_counts[shelf_idx] = count
                            total_items += count
                            # Add values of top 'count' items from this shelf
                            total_value += sum(item[1] for item in shelf[:count])
                
                # Check if sequence is valid:
                # 1. Total items <= bag_size
                # 2. No equal counts between shelves
                # 3. Sequence is continuous (allowing one empty shelf between)
                if (total_items <= bag_size and 
                    len(set(shelf_counts.values())) == len(shelf_counts) and
                    is_valid_sequence(shelf_counts)):
                    max_value = max(max_value, total_value)
    print('CS',current_sequence) #tentatively
    print(shelf_counts)#tentatively
    return round(max_value, 1)

def is_valid_sequence(shelf_counts):
    # Check if sequence is continuous (allowing one empty shelf between)
    shelf_nums = sorted(shelf_counts.keys())
    for i in range(len(shelf_nums) - 1):
        if shelf_nums[i + 1] - shelf_nums[i] > 2:
            return False
    return True

n,k,bag_size = map(int,input().split())
items = []
for i in range(n):
    item_info = input().split()
    item_id = int(item_info[0])
    value = float(item_info[1])
    items.append((item_id, value))

print(solve_supermarket_problem(n, k, bag_size, items))