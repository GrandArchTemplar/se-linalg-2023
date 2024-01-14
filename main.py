import re


def read_matrix(input_lines, size):
    n, m = size
    result = []
    for h in range(m):
        result.append([complex(num) for num in input_lines.pop(0).split()])
    return result


def read_vector(input_lines):
    return [complex(num) for num in input_lines.pop(0).strip().split()]


def parse(input_lines):
    expression = input_lines.pop(0)
    matrix = {}
    vectors = {}

    for matrix_name in re.findall(r'[A-Z]', expression):
        if matrix_name not in matrix.keys():
            dims = list(map(int, input_lines.pop(0).split()))
            if len(dims) == 1:
                dims = [dims[0], dims[0]]
            matrix[matrix_name] = read_matrix(input_lines, dims)

    for vector_name in re.findall(r'[a-wyz]', expression):
        if vector_name not in vectors.keys():
            input_lines.pop(0)
            vectors[vector_name] = read_vector(input_lines)
    return expression, matrix, vectors


def dot_product(matrix, vector):
    return [sum(m * v for m, v in zip(matrix_row, vector)) for matrix_row in matrix]


def add_matrices(A, B):
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def scalar_multiply(matrix, scalar):
    return [[element * scalar for element in row] for row in matrix]


def prepare_side(side: str, variables: dict):
    equation_parts = re.split(r"(\+|\-)", side)

    A = None
    b = None
    sign = 1

    for part in equation_parts:
        if part in '+-':
            sign = -1 if part == '-' else 1
            continue
        tmp = None
        for char in part:
            if char != "x":
                if tmp is None:
                    tmp = variables[char]
                else:
                    tmp = dot_product(tmp, variables[char])
        if "x" in part:
            tmp = scalar_multiply(tmp, sign)
            A = tmp if A is None else add_matrices(A, tmp)
        else:
            tmp = [t * sign for t in tmp]
            b = tmp if b is None else [sum(x) for x in zip(b, tmp)]

    return A, b


def transform(expression, matrix, vectors):
    left_side, right_side = expression.split('=')
    variables = {}
    variables.update(matrix)
    variables.update(vectors)
    try:
        A, b = prepare_side(left_side, variables)
        if right_side != "0":
            C, d = prepare_side(right_side, variables)
            if C is not None:
                A = add_matrices(A, scalar_multiply(C, -1))
            if b is not None and d is not None:
                b = [bi - di for bi, di in zip(d, b)]
        else:
            b = [-bi for bi in b]
        if A is None or b is None:
            return "нет числа"
        else:
            return A, b
    except Exception as e:
        return "нет числа"

def solve_gaussian(A, b):
    n = len(A)
    if len(b) != n:
        raise ValueError("Неверный ввод. A & b.", len(b), n)
    for k in range(n - 1):
        maxindex = k + max(range(k, n), key=lambda x: abs(A[x][k]))
        if A[maxindex][k] == 0:
            raise ValueError("Матрица является единственной.")
        if maxindex != k:
            A[k], A[maxindex] = A[maxindex], A[k]
            b[k], b[maxindex] = b[maxindex], b[k]
            if A[k][k] == 0:
                raise ValueError("Пустая матрица")

        for row in range(k + 1, n):
            multiplier = A[row][k] / A[k][k]
            A[row][k:] = [ar - multiplier * ak for ar, ak in zip(A[row][k:], A[k][k:])]
            b[row] = b[row] - multiplier * b[k]
    x = [0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (b[k] - sum(A[k][j] * x[j] for j in range(k + 1, n))) / A[k][k]
    return x


file_name = "input.txt"
with open(file_name, 'r') as file:
    input_lines = file.read().splitlines()
expression, matrix, vectors = parse(input_lines)
A, b = transform(expression, matrix, vectors)
x = solve_gaussian(A.copy(), b.copy())
result = '\n'.join(f'x{i+1} = {str(elem)[1:-4]}' for i, elem in enumerate(x))
print(result)
