clear; clc;

% Parameters
N = 5;
T = 10^6;
p = [1/2, 1/2];
s = ones(N, 1);
num_graphs = 21;

sim_all_sharer_performance_n_5 = zeros(1, num_graphs);
sim_all_withholder_performance_n_5 = zeros(1, num_graphs);

% Load data
data = load('n_5.mat');

for i = 1:num_graphs
    graph_name = sprintf('graph_%d', i);
    Gr = double(data.(graph_name));
    
    % Compute sharer and withholder performance for the current graph
    [sim_all_sharer_performance_n_5(i), sim_all_withholder_performance_n_5(i)] = average_error_parallel(Gr, s, p, T);
end

% Save results
save('sim_full_performance_n_5.mat', 'sim_all_sharer_performance_n_5', 'sim_all_withholder_performance_n_5');