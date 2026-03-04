class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0

if __name__ == '__main__':
    my_counter = Counter()
    for _ in range(5):
        my_counter.increment()

    print(f'total count: {my_counter.get_count()}')