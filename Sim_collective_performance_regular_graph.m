clear; clc;

% Parameters
N = 50;
T = 10^6;
p = [1/2, 1/2];
s = ones(N, 1);
k_array = 2:12;
num_graphs = length(k_array);

sim_s_perf_array = zeros(1, num_graphs);
sim_w_perf_array = zeros(1, num_graphs);
collective_n50_simu = zeros(1, num_graphs);

% Load data
data = load('RR_n50_k2_12.mat');

for i = 1:num_graphs
    if iscell(data.adj_matrices_RR_n50)
        Gr = full(double(data.adj_matrices_RR_n50{i}));
    else
        Gr = full(double(data.adj_matrices_RR_n50(:, :, i)));
    end
    
    % Compute sharer and withholder performance
    [sim_s_perf, sim_w_perf] = average_error_parallel(Gr, s, p, T);
    
    sim_s_perf_array(i) = sim_s_perf;
    sim_w_perf_array(i) = sim_w_perf;
    
    prob_c = data.rhoc_array_RR_n50(i);
    prob_d = data.rhod_array_RR_n50(i);
    
    weight_c = prob_c / (prob_c + prob_d);
    weight_d = prob_d / (prob_c + prob_d);
    collective_n50_simu(i) = weight_c * sim_s_perf + weight_d * sim_w_perf;
end

% Save results
save('final_collective_RR_n50_simu.mat', 'k_array', 'sim_s_perf_array', 'sim_w_perf_array', 'collective_n50_simu');