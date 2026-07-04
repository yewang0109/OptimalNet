# Simulation for strategy evolution on random regular networks of size N=50
# Calculate fixation probabilities for degrees k = 2, ..., 12

import numpy as np
from numba import njit
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import scipy.io as scio

# Parameters
N = 50
G = 10 ** 6
n_avg = 5 * 10 ** 6
n_batches = 20
t_array = np.ones(N, dtype=np.float64)
c_array = np.ones(N, dtype=np.float64)
p_array = np.array([0.5, 0.5], dtype=np.float64)
delta = 0.002
R = 3

pool = ProcessPoolExecutor()


@njit
def f_node_neighbor(matrix, num):
    """Build neighbor lists and node degrees from an adjacency matrix."""
    number_record = np.zeros(num, dtype=np.int32)
    neighbor_record = np.zeros((num, num), dtype=np.int32) - 1
    for i in range(num):
        for j in range(i, num):
            if matrix[i, j] > 0:
                neighbor_record[i, number_record[i]] = j
                neighbor_record[j, number_record[j]] = i
                number_record[i] += 1
                number_record[j] += 1
    return neighbor_record, number_record


@njit
def f_strategy_generate_cooperator(num):
    """Initial state: one cooperator and all others defectors."""
    strategy = np.zeros(num, dtype=np.int8)
    index = np.random.randint(0, num)
    strategy[index] = 1
    return strategy


@njit
def f_strategy_generate_defector(num):
    """Initial state: one defector and all others cooperators."""
    strategy = np.ones(num, dtype=np.int8)
    index = np.random.randint(0, num)
    strategy[index] = 0
    return strategy


@njit
def f_cal_payoff(id_x, matrix, strategy, t_array, c_array, p_array, R, num):
    """Calculate payoff of node id_x."""
    denominator_inv = 1.0 / (1.0 - np.sum(p_array * p_array))
    payoff = R * t_array[id_x] * denominator_inv
    s_x = strategy[id_x]
    for id_y in range(num):
        s_y = strategy[id_y]
        payoff += matrix[id_x, id_y] * (R * s_y * t_array[id_y] * denominator_inv - c_array[id_x] * s_x)
    return payoff


@njit
def f_roulettewheelselection(seq):
    length = len(seq)
    rand = np.random.random()
    front = seq[0]
    back = 0
    if length == 1:
        return 0
    for i in range(length - 1):
        if rand >= back and rand <= front:
            return i
        else:
            back += seq[i]
            front += seq[i + 1]
    return i + 1


@njit
def f_death_birth_process(strategy, matrix, neighbor_record, number_record, t_array, c_array, p_array, R, num):
    active_node_record = np.arange(num)
    death = np.random.choice(active_node_record)
    neighbor_of_death = neighbor_record[death, 0:number_record[death]]
    fitness_array = np.zeros(number_record[death])
    tik = 0
    for index in neighbor_of_death:
        fitness_array[tik] = matrix[death, index] * np.exp(
            delta * f_cal_payoff(index, matrix, strategy, t_array, c_array, p_array, R, num))
        tik += 1
    prob_array = fitness_array / np.sum(fitness_array)
    birth = neighbor_of_death[f_roulettewheelselection(prob_array)]
    strategy[death] = strategy[birth]
    return strategy


@njit
def f_single_turn_cooperator(neighbor_record, number_record, mat, t_array, c_array, p_array, R, num):
    """One run starting from a single cooperator."""
    strategy = f_strategy_generate_cooperator(num)
    tik1 = 0
    while tik1 < G:
        strategy = f_death_birth_process(strategy, mat, neighbor_record, number_record, t_array, c_array, p_array, R,
                                         num)
        sum_strategy = np.sum(strategy)
        if sum_strategy == 0:
            return 0
        if sum_strategy == num:
            return 1
        tik1 += 1
    return 0


@njit
def f_single_turn_defector(neighbor_record, number_record, mat, t_array, c_array, p_array, R, num):
    """One run starting from a single defector."""
    strategy = f_strategy_generate_defector(num)
    tik1 = 0
    while tik1 < G:
        strategy = f_death_birth_process(strategy, mat, neighbor_record, number_record, t_array, c_array, p_array, R,
                                         num)
        sum_strategy = np.sum(strategy)
        if sum_strategy == 0:
            return 1
        if sum_strategy == num:
            return 0
        tik1 += 1
    return 0


@njit
def f_avg_return_cooperator(neighbor_record, number_record, static_adj_mat, n_avg, t_array, c_array, p_array, R, num):
    """Average fixation result over n_avg cooperator runs."""
    val = 0
    for i in range(n_avg):
        val += f_single_turn_cooperator(neighbor_record, number_record, static_adj_mat, t_array, c_array, p_array, R,
                                        num)
    return val / n_avg


@njit
def f_avg_return_defector(neighbor_record, number_record, static_adj_mat, n_avg, t_array, c_array, p_array, R, num):
    """Average fixation result over n_avg defector runs."""
    val = 0
    for i in range(n_avg):
        val += f_single_turn_defector(neighbor_record, number_record, static_adj_mat, t_array, c_array, p_array, R, num)
    return val / n_avg


def main(t_array, c_array, p_array, R, Gp):
    static_adj_mat_origin = nx.adjacency_matrix(Gp).todense()
    num, _ = static_adj_mat_origin.shape
    neighbor_record, number_record = f_node_neighbor(static_adj_mat_origin, num)

    neighbor_record_list = [neighbor_record] * n_batches
    number_record_list = [number_record] * n_batches
    static_adj_mat_list = [static_adj_mat_origin] * n_batches
    n_avg_list = [n_avg] * n_batches
    t_array_list = [t_array] * n_batches
    c_array_list = [c_array] * n_batches
    p_array_list = [p_array] * n_batches
    R_list = [R] * n_batches
    num_list = [num] * n_batches

    result_list_cooperator = list(
        pool.map(f_avg_return_cooperator, neighbor_record_list, number_record_list, static_adj_mat_list, n_avg_list,
                 t_array_list, c_array_list, p_array_list, R_list, num_list))
    result_list_defector = list(
        pool.map(f_avg_return_defector, neighbor_record_list, number_record_list, static_adj_mat_list, n_avg_list,
                 t_array_list, c_array_list, p_array_list, R_list, num_list))

    return result_list_cooperator, result_list_defector


if __name__ == "__main__":
    k_array = np.arange(2, 13)
    length = len(k_array)

    simu_arr_cooperator = np.zeros(length)
    simu_arr_defector = np.zeros(length)

    adj_matrices = []

    for i in range(length):
        k = k_array[i]

        while True:
            Gp = nx.random_regular_graph(k, N)
            if nx.is_connected(Gp):
                break

        adj_matrices.append(nx.adjacency_matrix(Gp).todense())

        result_list_cooperator, result_list_defector = main(t_array, c_array, p_array, R, Gp)

        simu_arr_cooperator[i] = np.mean(result_list_cooperator)
        simu_arr_defector[i] = np.mean(result_list_defector)

    print(k_array)
    print(simu_arr_cooperator, simu_arr_defector)

    scio.savemat("RR_n50_k2_12.mat", {
        'k_array_RR_n50': k_array,
        'rhoc_array_RR_n50': simu_arr_cooperator,
        'rhod_array_RR_n50': simu_arr_defector,
        'adj_matrices_RR_n50': np.stack(adj_matrices, axis=-1)  # (50, 50, 11)
    })
