import time

def log_message(message, log_file):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def binary_search_recursive(func, interval, tol=1e-5, log_file='log.txt'):
    start_time = time.time()
    a, b = interval
    fa, fb = func(a), func(b)

    log_message(f"Интервал: [{a}, {b}], f(a): {fa}, f(b): {fb}", log_file)

    if fa * fb > 0:
        log_message("Нет корней в интервале.", log_file)
        return None

    mid = (a + b) / 2
    fmid = func(mid)

    log_message(f"Средняя точка: {mid}, f(mid): {fmid}, прошедшее время: {time.time() - start_time:.6f} секунд", log_file)

    if abs(fmid) < tol:
        return mid

    if fa * fmid < 0:
        return binary_search_recursive(func, (a, mid), tol, log_file)
    else:
        return binary_search_recursive(func, (mid, b), tol, log_file)

def binary_search_iterative(func, interval, tol=1e-5, log_file='log.txt'):
    start_time = time.time()
    a, b = interval
    fa, fb = func(a), func(b)

    log_message(f"Интервал: [{a}, {b}], f(a): {fa}, f(b): {fb}", log_file)

    if fa * fb > 0:
        log_message("Нет корней в интервале.", log_file)
        return None

    while True:
        mid = (a + b) / 2
        fmid = func(mid)

        log_message(f"Средняя точка: {mid}, f(mid): {fmid}, прошедшее время: {time.time() - start_time:.6f} секунд", log_file)

        if abs(fmid) < tol:
            return mid

        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

def example_function(x):
    return x**2 - 2  

interval = [0, 2]

log_file = 'binary_search_log.txt'

with open(log_file, 'w') as f:
    f.write("Binary Search Log\n")

recursive_result = binary_search_recursive(example_function, interval, log_file=log_file)
print("Результат рекурсивного бинарного поиска:", recursive_result)

iterative_result = binary_search_iterative(example_function, interval, log_file=log_file)
print("Результат итерального бинарного поиска:", iterative_result)
