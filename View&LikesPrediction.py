import numpy as np

watcher_train = np.array([5.800,8.756,5.700,4.100,4.600,2.200])
likes_train = np.array([27.5,54.9,34.2,23.0,20.3,30.1])


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, cost_function, gradient_function, num_iters):
    j_history = []
    w_b_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        j_history.append(cost_function(x, y, w, b))
        w_b_history.append([w,b])

    return w, b, j_history, w_b_history


"""  initialization  """

w_final, b_final, j_hist, w_b_hist = gradient_descent(watcher_train, likes_train,
                                                      0, 0, 0.01, compute_cost,
                                                      compute_gradient, 100000)

print("The number of likes in millions for the song, which has been watched 8.367 billion times, is as follows: ", w_final * 8.367 + b_final)
