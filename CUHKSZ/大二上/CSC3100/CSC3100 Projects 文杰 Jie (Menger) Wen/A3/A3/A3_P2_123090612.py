n, k, bag_size = map(int, input().split())
n = int(n)
k = int(k)
bag_size = int(bag_size)

# List to store item data and track maximum decimal places
items_data = []
max_decimal_places = 0

# Read items and determine the maximum number of decimal places
for _ in range(n):
    id_str, value_str = input().split()
    id = int(id_str)
    # Determine the number of decimal places
    if '.' in value_str:
        integer_part, decimal_part = value_str.split('.')
        num_decimal_places = len(decimal_part)
    else:
        integer_part = value_str
        decimal_part = ''
        num_decimal_places = 0
    # Update maximum decimal places
    if num_decimal_places > max_decimal_places:
        max_decimal_places = num_decimal_places
    # Store item data
    shelf_num = id % k
    items_data.append({
        'id': id,
        'value_str': value_str,
        'integer_part': integer_part,
        'decimal_part': decimal_part,
        'num_decimal_places': num_decimal_places,
        'shelf': shelf_num
    })

# Convert item values to scaled integers
for item in items_data:
    # Pad decimal part with zeros to match maximum decimal places
    decimal_part_padded = item['decimal_part'].ljust(max_decimal_places, '0')
    value_int_str = item['integer_part'] + decimal_part_padded
    item['value_int'] = int(value_int_str)

# Assign items to shelves
shelves = [[] for _ in range(k)]
for item in items_data:
    shelves[item['shelf']].append(item)

# Sort items on each shelf in descending order of IDs
for shelf in shelves:
    shelf.sort(key=lambda x: -x['id'])

# Build the items array in order of shelves in the ring
items_array = []
shelf_indices = []
for i in range(k):
    shelf = shelves[i]
    for item in shelf:
        items_array.append(item)
        shelf_indices.append(i)

# Extend the items array to simulate the circular nature
items_array_extended = items_array + items_array

max_total_value_int = 0  # Initialize maximum total value as integer

# Precompute whether shelves have items
shelf_has_items = [len(shelves[i]) > 0 for i in range(k)]

# Function to format the total value for output
def format_total_value(total_value_int, max_decimal_places):
    total_value_str = str(total_value_int)
    if max_decimal_places == 0:
        return total_value_str
    # Ensure the string has enough digits
    if len(total_value_str) <= max_decimal_places:
        total_value_str = total_value_str.rjust(max_decimal_places + 1, '0')
    integer_part = total_value_str[:-max_decimal_places]
    decimal_part = total_value_str[-max_decimal_places:]
    return integer_part + '.' + decimal_part

# Main loop to find the maximum total value
for start in range(n):
    shelves_taken = {}
    total_items_taken = 0
    total_value_int = 0
    empty_shelf_skipped = False
    prev_shelf = None
    i = start

    while total_items_taken < bag_size and i < start + n:
        item = items_array_extended[i]
        shelf = item['shelf']

        # Check for empty shelves between the current and previous shelf
        if prev_shelf is not None:
            dist = (shelf - prev_shelf) % k
            if dist != 0:
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
        total_value_int += item['value_int']
        i += 1

        # Check if the counts of items from each shelf are unique
        counts_list = list(shelves_taken.values())
        if len(counts_list) == len(set(counts_list)):
            if total_value_int > max_total_value_int:
                max_total_value_int = total_value_int

# Output the maximum total value formatted correctly
print(format_total_value(max_total_value_int, max_decimal_places))