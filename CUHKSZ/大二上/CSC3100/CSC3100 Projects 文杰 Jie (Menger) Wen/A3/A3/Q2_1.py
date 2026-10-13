n, k, bag_size = map(int, input().split())
n = int(n)
k = int(k)
bag_size = int(bag_size)

# Read items and assign to shelves
shelves = [[] for _ in range(k)]
for _ in range(n):
    id_str, value_str = input().split()
    id = int(id_str)
    value = float(value_str)
    shelf_num = id % k
    shelves[shelf_num].append({'id': id, 'value': value, 'shelf': shelf_num})

# Sort items on each shelf in descending order of IDs
for shelf in shelves:
    shelf.sort(key=lambda x: -x['id'])

# Build the items array in order of shelves in the ring
items_array = []
shelf_indices = []
for i in range(k):
    shelf_num = i
    shelf = shelves[shelf_num]
    for item in shelf:
        items_array.append(item)
        shelf_indices.append(shelf_num)

# Extend the items array to simulate the circular nature
items_array *= 2
shelf_indices *= 2

max_total_value = 0.0

# Precompute empty shelves
shelf_has_items = [len(shelves[i]) > 0 for i in range(k)]

for start in range(n):
    shelves_taken = {} # Count of items taken from each shelf
    counts = {} # Determine whether the number of items taken from each shelf is unique
    total_items = 0 # Total number of items taken
    total_value = 0.0 # Total value of items taken
    empty_shelf_skipped = False
    prev_shelf = None
    i = start
    while total_items < bag_size and i < start + n:
        item = items_array[i]
        shelf = item['shelf']

        # Check if moving from prev_shelf to shelf skips empty shelves
        if prev_shelf is not None:
            # Compute distance between shelves in the ring
            dist = (shelf - prev_shelf) % k
            if dist == 0:
                pass  # Same shelf
            else:
                empty_shelves_between = 0
                for s in range(1, dist):
                    inter_shelf = (prev_shelf + s) % k
                    if not shelf_has_items[inter_shelf]:
                        empty_shelves_between +=1
                if empty_shelves_between > 1 or (empty_shelf_skipped and empty_shelves_between >=1):
                    break
                if empty_shelves_between >=1:
                    empty_shelf_skipped = True
        prev_shelf = shelf

        shelves_taken[shelf] = shelves_taken.get(shelf, 0) +1
        count = shelves_taken[shelf]
        if count in counts:
            break  # Duplicate count of items from different shelves
        counts[count] = shelf
        total_items +=1
        total_value += item['value']
        if total_items > bag_size:
            break
        if total_value > max_total_value:
            max_total_value = total_value
        i +=1

print(max_total_value)