# 用另一个队列实现对原队列的翻转
class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)

    def reverse(self):
        new_queue = Queue()
        while not self.is_empty():
            new_queue.enqueue(self.dequeue())
        return new_queue

# 测试
q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
q.enqueue(5)
print(q.items)
q_reverse = q.reverse()
print(q_reverse.items)