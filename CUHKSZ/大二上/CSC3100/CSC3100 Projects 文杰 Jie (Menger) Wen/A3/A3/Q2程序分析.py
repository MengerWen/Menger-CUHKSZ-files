'''
用以下Sample input为例
6 4 3
15 23.6
23 3.6
198 4.2
7 5.3
9 15.6
1453 31.1
'''

n = 6
k = 4
bag_size = 3

shelves = [
    [],  # 货架 0 为空
    [{'id': 1453, 'value': 31.1, 'shelf': 1}, {'id': 9, 'value': 15.6, 'shelf': 1}],  # 货架 1
    [{'id': 198, 'value': 4.2, 'shelf': 2}],  # 货架 2
    [{'id': 23, 'value': 3.6, 'shelf': 3}, {'id': 15, 'value': 23.6, 'shelf': 3}, \
        {'id': 7, 'value': 5.3, 'shelf': 3}]  # 货架 3
]

# 最初
items_array = [
    {'id': 1453, 'value': 31.1, 'shelf': 1},
    {'id': 9, 'value': 15.6, 'shelf': 1},
    {'id': 198, 'value': 4.2, 'shelf': 2},
    {'id': 23, 'value': 3.6, 'shelf': 3},
    {'id': 15, 'value': 23.6, 'shelf': 3},
    {'id': 7, 'value': 5.3, 'shelf': 3}
]
shelf_indices = [1, 1, 2, 3, 3, 3]

# ×2之后
items_array = [
    {'id': 1453, 'value': 31.1, 'shelf': 1},{'id': 9, 'value': 15.6, 'shelf': 1},
    {'id': 198, 'value': 4.2, 'shelf': 2},{'id': 23, 'value': 3.6, 'shelf': 3},
    {'id': 15, 'value': 23.6, 'shelf': 3},{'id': 7, 'value': 5.3, 'shelf': 3},

    {'id': 1453, 'value': 31.1, 'shelf': 1},{'id': 9, 'value': 15.6, 'shelf': 1},
    {'id': 198, 'value': 4.2, 'shelf': 2},{'id': 23, 'value': 3.6, 'shelf': 3},
    {'id': 15, 'value': 23.6, 'shelf': 3},{'id': 7, 'value': 5.3, 'shelf': 3},
]
shelf_indices = [1, 1, 2, 3, 3, 3, 1, 1, 2, 3, 3, 3]

# 第一次循环
start = 0
empty_shelf_skipped = False
prev_shelf = None
i = 0

total_items = 0
start + n = 6
total_items < bag_size, i < start + n
item = {'id': 1453, 'value': 31.1, 'shelf': 1}
shelf = 1

prev_shelf is None # 跳过这个判断

prev_shelf = 1

shelves_taken[1] = shelves_taken.get(1, 0) +1
shelves_taken = {1: 1}
count = 1
counts = {1: 1}
total_items = 1
total_value = 31.1

# 第二次循环
i = 1
item = {'id': 9, 'value': 15.6, 'shelf': 1}
shelf = 1

shelves_taken = {1: 2}
count = 2
counts = {1: 1, 2: 1}
total_items = 2
total_value = 46.7

# 第三次循环
i = 2
item = {'id': 198, 'value': 4.2, 'shelf': 2}
shelf = 2

dist = 1
empty_shelves_between = 0

prev_shelf = 2

shelves_taken = {1: 2, 2: 1}
count = 1
counts = {1: 1, 2: 1}
total_items = 3
total_value = 50.9