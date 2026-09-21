x = [4, 8, 15, 16, 23, 42]

print(x[2])
print(x[-2])
print(x[1:4])
print(x[::2])
print(x[::-1])

numbers = [3, 12, 7, 20, 4, 15, 1]
L = [val for val in numbers if 5 < val < 18]

def mean(values):
    return sum(values) / len(values)

def add_vector(a, b):
    l = []
    for x, y in zip(a, b):
        l.append(x + y)
    return l

def dot(a, b):
    r = 0
    for x, y in zip(a, b):
        r += x * y
    return r

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

def sum_rows(matrix):
    r = []
    for row in matrix:
        total = 0
        for value in row:
            total += value
        r.append(total)
    return r

def sum_rows2(matrix):
    return [sum(row) for row in matrix]

def sum_columns(matrix):
    result = [0] * len(matrix[0])
    for row in matrix:
        for j in range(len(row)):
            result[j] += row[j]
    return result

def sum_columns2(matrix):
    return [sum(columm) for columm in zip(*matrix)]

# zip(*matrix) => transform [1,2,3],    into (1, 4, 7),
#                           [4,5,6],         (2, 5, 8),
#                           [7,8,9]          (3, 6, 9)

print(sum_rows(matrix))
print(sum_columns(matrix))

y = [3, 7, 2, 9, 4]

def normalize(values):
    m = sum(values) / len(values)
    r = [x - m for x in values]
    return r
print(normalize(y))