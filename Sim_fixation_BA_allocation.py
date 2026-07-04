# Calculate fixation probabilities under uniform, proportional, and
# inverse sample-allocation schemes for the network specified by NETWORK_TYPE

import numpy as np
from numba import njit
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import scipy.io as scio

# Change this to "BA", "WS", "ER", or "RR" to switch network type.
# Requires the matching network_{type}_n100_k4.mat file in the same folder,
# e.g. network_ba_n100_k4.mat for NETWORK_TYPE = "BA".
NETWORK_TYPE = "BA"

# Parameters
N = 100
G = 10 ** 6
n_avg = 5 * 10 ** 6
n_batches = 20
delta = 0.002
p_array = np.array([0.5, 0.5], dtype=np.float64)

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
def f_death_birth_process(strategy, matrix, neighbor_record, number_record,
                           t_array, c_array, p_array, R, delta, num):
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
def f_single_turn_cooperator(neighbor_record, number_record, mat, t_array, c_array, p_array, R, delta, num):
    """One run starting from a single cooperator."""
    strategy = f_strategy_generate_cooperator(num)
    tik1 = 0
    while tik1 < G:
        strategy = f_death_birth_process(strategy, mat, neighbor_record, number_record,
                                          t_array, c_array, p_array, R, delta, num)
        sum_strategy = np.sum(strategy)
        if sum_strategy == 0:
            return 0
        if sum_strategy == num:
            return 1
        tik1 += 1
    return 0


@njit
def f_single_turn_defector(neighbor_record, number_record, mat, t_array, c_array, p_array, R, delta, num):
    """One run starting from a single defector."""
    strategy = f_strategy_generate_defector(num)
    tik1 = 0
    while tik1 < G:
        strategy = f_death_birth_process(strategy, mat, neighbor_record, number_record,
                                          t_array, c_array, p_array, R, delta, num)
        sum_strategy = np.sum(strategy)
        if sum_strategy == 0:
            return 1
        if sum_strategy == num:
            return 0
        tik1 += 1
    return 0


@njit
def f_avg_return_cooperator(neighbor_record, number_record, static_adj_mat, n_avg,
                             t_array, c_array, p_array, R, delta, num):
    """Average fixation result over n_avg cooperator runs."""
    val = 0
    for i in range(n_avg):
        val += f_single_turn_cooperator(neighbor_record, number_record, static_adj_mat,
                                         t_array, c_array, p_array, R, delta, num)
    return val / n_avg


@njit
def f_avg_return_defector(neighbor_record, number_record, static_adj_mat, n_avg,
                           t_array, c_array, p_array, R, delta, num):
    """Average fixation result over n_avg defector runs."""
    val = 0
    for i in range(n_avg):
        val += f_single_turn_defector(neighbor_record, number_record, static_adj_mat,
                                       t_array, c_array, p_array, R, delta, num)
    return val / n_avg


@njit
def allocate_resources_njit(degree_sequence, alloc_type_code):
    """Largest-remainder allocation of samples across nodes.
    alloc_type_code: 0 = uniform, 1 = proportional, 2 = inverse (to degree)."""
    N = len(degree_sequence)
    D_total = int(np.sum(degree_sequence))

    inv_weights = np.empty(N)
    for i in range(N):
        if alloc_type_code == 0:
            inv_weights[i] = 1.0
        elif alloc_type_code == 1:
            inv_weights[i] = degree_sequence[i]
        elif alloc_type_code == 2:
            if degree_sequence[i] == 0:
                inv_weights[i] = 0.0
            else:
                inv_weights[i] = 1.0 / degree_sequence[i]

    rand_idx = np.arange(N)
    for i in range(N - 1, 0, -1):
        j = np.random.randint(i + 1)
        rand_idx[i], rand_idx[j] = rand_idx[j], rand_idx[i]
    inv_weights_shuffled = inv_weights[rand_idx]

    total_weight = np.sum(inv_weights_shuffled)
    T_float = inv_weights_shuffled / total_weight * D_total
    T_int = np.floor(T_float).astype(np.int64)
    remainder = D_total - np.sum(T_int)

    frac_part = T_float - T_int
    idx_sort = np.arange(N)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if frac_part[idx_sort[i]] < frac_part[idx_sort[j]]:
                idx_sort[i], idx_sort[j] = idx_sort[j], idx_sort[i]

    for i in range(remainder):
        T_int[idx_sort[i]] += 1

    T = np.empty(N, dtype=np.int64)
    for i in range(N):
        T[rand_idx[i]] = T_int[i]

    while True:
        zero_nodes = np.empty(N, dtype=np.int64)
        n_zero = 0
        for i in range(N):
            if T[i] == 0:
                zero_nodes[n_zero] = i
                n_zero += 1
        if n_zero == 0:
            break

        donor_candidates = np.empty(N, dtype=np.int64)
        n_donors = 0
        for j in range(N):
            if T[j] > 1:
                donor_candidates[n_donors] = j
                n_donors += 1
        if n_donors == 0:
            break

        for k in range(n_zero):
            if n_donors == 0:
                break
            pick = np.random.randint(n_donors)
            donor = donor_candidates[pick]
            if T[donor] > 1:
                T[donor] -= 1
                T[zero_nodes[k]] += 1
    return T


def main(t_array, c_array, p_array, R, delta, Gp):
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
    delta_list = [delta] * n_batches
    num_list = [num] * n_batches

    result_list_cooperator = list(
        pool.map(f_avg_return_cooperator, neighbor_record_list, number_record_list, static_adj_mat_list,
                 n_avg_list, t_array_list, c_array_list, p_array_list, R_list, delta_list, num_list))
    result_list_defector = list(
        pool.map(f_avg_return_defector, neighbor_record_list, number_record_list, static_adj_mat_list,
                 n_avg_list, t_array_list, c_array_list, p_array_list, R_list, delta_list, num_list))

    return result_list_cooperator, result_list_defector


def run_allocation_scheme(alloc_code, alloc_name, degree_seq, adj_mat, Gp, R_array, network_type):
    """Run the fixation-probability pipeline for a single allocation scheme
    and save the results."""
    t_array = allocate_resources_njit(degree_seq, alloc_code)
    c_array = t_array

    length = len(R_array)
    simu_arr_cooperator = np.zeros(length)
    simu_arr_defector = np.zeros(length)

    for i in range(length):
        R = R_array[i]
        result_list_cooperator, result_list_defector = main(t_array, c_array, p_array, R, delta, Gp)
        simu_arr_cooperator[i] = np.mean(result_list_cooperator)
        simu_arr_defector[i] = np.mean(result_list_defector)

    print(f"--- {network_type}, {alloc_name} ---")
    print("R_array:", R_array)
    print("rho_c:", simu_arr_cooperator)
    print("rho_d:", simu_arr_defector)
    print()

    scio.savemat(f"{network_type}_n100_k4_sample_allocation_{alloc_name}.mat", {
        f"R_array_{network_type}_k4_sample_allocation_{alloc_name}": R_array,
        f"rhoc_array_{network_type}_k4_sample_allocation_{alloc_name}": simu_arr_cooperator,
        f"rhod_array_{network_type}_k4_sample_allocation_{alloc_name}": simu_arr_defector,
        f"adjacency_matrix_{network_type}_k4_sample_allocation_{alloc_name}": np.array(adj_mat),
        f"t_array_{network_type}_k4_sample_allocation_{alloc_name}": np.array(t_array),
    })


if __name__ == "__main__":
    R_array = np.round(np.linspace(1, 3, 11), 1)

    data = scio.loadmat(f"network_{NETWORK_TYPE.lower()}_n100_k4.mat")
    edge_array = data['edge_array']
    Gp = nx.Graph()
    Gp.add_edges_from(edge_array)
    adj_mat = nx.adjacency_matrix(Gp).todense()
    degree_seq = np.array(adj_mat.sum(axis=1), dtype=np.float64).flatten()

    allocation_schemes = [
        (0, "uniform"),
        (1, "proportional"),
        (2, "inverse"),
    ]

    for alloc_code, alloc_name in allocation_schemes:
        run_allocation_scheme(alloc_code, alloc_name, degree_seq, adj_mat, Gp, R_array, NETWORK_TYPE)