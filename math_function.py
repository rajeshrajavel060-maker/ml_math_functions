import math


def validate_data(data):
    if not data:
        raise ValueError("Data cannot be empty")



def calculate_mean(numbers):
    validate_data(numbers)
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)


def calculate_variance(numbers):
    validate_data(numbers)
    avg = calculate_mean(numbers)

    total = 0
    for num in numbers:
        total += (num - avg) ** 2

    return total / len(numbers)


def calculate_standard_deviation(numbers):
    var = calculate_variance(numbers)
    return math.sqrt(var)


def calculate_covariance(x, y):
    if len(x) != len(y):
        raise ValueError("Lists must have same length")

    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)

    total = 0
    for i in range(len(x)):
        total += (x[i] - mean_x) * (y[i] - mean_y)

    return total / len(x)


def calculate_correlation(x, y):
    cov = calculate_covariance(x, y)

    std_x = calculate_standard_deviation(x)
    std_y = calculate_standard_deviation(y)

    if std_x == 0 or std_y == 0:
        return 0

    return cov / (std_x * std_y)



def calculate_z_score(value, data):
    mean_value = calculate_mean(data)
    std_value = calculate_standard_deviation(data)

    if std_value == 0:
        return 0

    return (value - mean_value) / std_value




def find_min(data):
    minimum = data[0]
    for v in data:
        if v < minimum:
            minimum = v
    return minimum


def find_max(data):
    maximum = data[0]
    for v in data:
        if v > maximum:
            maximum = v
    return maximum


def min_max_scale(data):
    validate_data(data)

    minimum = find_min(data)
    maximum = find_max(data)

    scaled = []

    for value in data:
        if maximum == minimum:
            scaled.append(0)
        else:
            scaled.append((value - minimum) / (maximum - minimum))

    return scaled



def basic_probability(success, total):
    if total == 0:
        raise ValueError("Total cannot be zero")
    return success / total


def apply_bayes(p_b_given_a, p_a, p_b):
    if p_b == 0:
        return 0
    return (p_b_given_a * p_a) / p_b



def multiply_matrices(A, B):

    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


def transpose(matrix):

    rows = len(matrix)
    cols = len(matrix[0])

    result = []

    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        result.append(row)

    return result



def simple_linear_prediction(x, intercept, slope):
    return intercept + slope * x


def calculate_mse(actual, predicted):

    if len(actual) != len(predicted):
        raise ValueError("Lists must be same length")

    total = 0

    for i in range(len(actual)):
        total += (actual[i] - predicted[i]) ** 2

    return total / len(actual)




def logistic_sigmoid(value):
    return 1 / (1 + math.exp(-value))



def compute_accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    if total == 0:
        return 0

    return (tp + tn) / total


def compute_precision(tp, fp):

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def compute_recall(tp, fn):

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def compute_f1(tp, fp, fn):

    p = compute_precision(tp, fp)
    r = compute_recall(tp, fn)

    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)



def calculate_entropy(prob_list):

    total_entropy = 0

    for p in prob_list:
        if p > 0:
            total_entropy += -p * math.log2(p)

    return total_entropy




if __name__ == "__main__":

    data = [5, 10, 15, 20, 25]
    x = [1, 2, 3, 4, 5]
    y = [3, 6, 9, 12, 15]

    print("Mean:", calculate_mean(data))
    print("Variance:", calculate_variance(data))
    print("Standard Deviation:", calculate_standard_deviation(data))
    print("Correlation:", calculate_correlation(x, y))
    print("Z Score (15):", calculate_z_score(15, data))
    print("Scaled Data:", min_max_scale(data))

    print("Probability:", basic_probability(4, 10))
    print("Bayes Result:", apply_bayes(0.7, 0.5, 0.6))

    A = [[2, 3], [4, 5]]
    B = [[1, 2], [3, 4]]

    print("Matrix Multiplication:", multiply_matrices(A, B))
    print("Transpose:", transpose(A))

    predictions = [simple_linear_prediction(i, 2, 3) for i in x]

    print("MSE:", calculate_mse(y, predictions))

    print("Sigmoid(2):", logistic_sigmoid(2))

    print("Accuracy:", compute_accuracy(50, 40, 5, 5))
    print("Precision:", compute_precision(50, 5))
    print("Recall:", compute_recall(50, 5))
    print("F1 Score:", compute_f1(50, 5, 5))

    probs = [0.6, 0.4]

    print("Entropy:", calculate_entropy(probs))