clear; clc;

% Parameters
N = 5;
T = ones(1, 5);
p = [1/2, 1/2];
c = ones(1, 5);
R = 6;
C = 1;
delta = 0.01;
complexity = 1 - sum(p .* p);
num_graphs = 21;

theory_collective_performance = zeros(1, num_graphs);
sim_collective_performance = zeros(1, num_graphs);
a = eye(N); 

% Load datasets
data = load('n_5.mat');
fixation_data = load('n_5_couple_effect.mat');
sim_results = load('sim_full_performance_n_5.mat');

for i = 1:num_graphs
    % Load the i-th graph
    graph_name = sprintf('graph_%d', i);
    Gr = double(data.(graph_name));
    num_nodes = size(Gr, 1);
    
    % Compute transition matrix and remeeting-time terms
    replace_mat = Gr;
    trans_mat = f_gen_trans_mat(replace_mat, num_nodes);
    retime2 = f_cal_remeet_time_two(trans_mat, num_nodes);
    retime3 = f_cal_remeet_time_three(trans_mat, retime2, num_nodes);
    [retime2_mat, ~] = f_reshape_retime(retime2, retime3, num_nodes); 
    
    rowSums = sum(Gr, 2);
    G_1 = Gr ./ rowSums;
    G_2 = G_1 * G_1;
    
    value_b = 0;
    value_c = 0;
    for j = 1:N
        for m = 1:N
            cont_b = 0;
            cont_c = 0;
            for l = 1:N
                cont_b = cont_b + Gr(l, m) * T(l) / complexity * retime2_mat(l, j);
                cont_c = cont_c + Gr(l, m) * c(m) * retime2_mat(m, j);
            end
            value_b = value_b + rowSums(j) / sum(rowSums) * (G_2(j, m) - a(j, m)) * cont_b;
            value_c = value_c + rowSums(j) / sum(rowSums) * (G_2(j, m) - a(j, m)) * cont_c;
        end
    end
    
    % Approximate fixation probabilities under weak selection
    rho_s = 1/N + delta/N * (R * value_b - C * value_c);
    rho_w = 1/N - delta/N * (R * value_b - C * value_c);
    
    % Analytical performance
    the_s_perf = 0;
    the_w_perf = 0;
    for m = 1:N
        the_s_perf = the_s_perf + complexity / (N * (T(m) + sum(T * Gr(:, m))));
        the_w_perf = the_w_perf + complexity / (N * T(m));
    end
    theory_collective_performance(i) = N/2 * rho_s * the_s_perf + N/2 * rho_w * the_w_perf;
    
    % Simulation performance
    prob_c = fixation_data.rhoc_array_n_5(i);
    prob_d = fixation_data.rhod_array_n_5(i);
    sim_s_perf = sim_results.sim_all_sharer_performance_n_5(i);
    sim_w_perf = sim_results.sim_all_withholder_performance_n_5(i);
    
    weight_c = prob_c / (prob_c + prob_d);
    weight_d = prob_d / (prob_c + prob_d);
    sim_collective_performance(i) = weight_c * sim_s_perf + weight_d * sim_w_perf;
end

save('final_collective_performance_n_5.mat', 'theory_collective_performance', 'sim_collective_performance');