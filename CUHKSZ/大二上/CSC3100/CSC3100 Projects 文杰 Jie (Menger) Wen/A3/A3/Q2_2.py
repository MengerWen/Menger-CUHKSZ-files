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

# Sort items on each shelf in descending order of IDs (or values)
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
items_array_extended = items_array + items_array  # Only need to double once
shelf_indices_extended = shelf_indices + shelf_indices

max_total_value = 0.0

# Precompute whether shelves have items
shelf_has_items = [len(shelves[i]) > 0 for i in range(k)]

# Total number of items in items_array
total_items = len(items_array)

for start in range(total_items):
    shelves_taken = {}
    total_items_taken = 0
    total_value = 0.0
    empty_shelf_skipped = False
    prev_shelf = None
    i = start
    skipped = False

    while total_items_taken < bag_size and i < start + total_items:
        item = items_array_extended[i]
        shelf = item['shelf']

        # Check if moving from prev_shelf to shelf skips empty shelves
        if prev_shelf is not None:
            dist = (shelf - prev_shelf) % k
            if dist == 0:
                pass  # Same shelf
            else:
                empty_shelves_between = 0
                for s in range(1, dist):
                    inter_shelf = (prev_shelf + s) % k
                    if not shelf_has_items[inter_shelf]:
                        empty_shelves_between += 1
                if empty_shelves_between > 1 or (empty_shelf_skipped and empty_shelves_between >= 1):
                    break  # Cannot skip more than one empty shelf
                if empty_shelves_between >= 1:
                    if empty_shelf_skipped:
                        break
                    empty_shelf_skipped = True
        prev_shelf = shelf

        shelves_taken[shelf] = shelves_taken.get(shelf, 0) + 1
        total_items_taken += 1
        total_value += item['value']
        if total_items_taken > bag_size:
            break
        i += 1

    # After collecting items, check if counts are unique
    counts_list = list(shelves_taken.values())
    if len(counts_list) == len(set(counts_list)):
        # Counts are unique
        if total_value > max_total_value:
            max_total_value = total_value

print(max_total_value)