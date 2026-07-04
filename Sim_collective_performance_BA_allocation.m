clear; clc;

% Parameters
network_type = 'BA';  % 'BA', 'WS', 'ER', or 'RR'
p = [1/2, 1/2];
T = 10^6;
schemes = {'uniform', 'proportional', 'inverse'};

for k = 1:length(schemes)
    scheme = schemes{k};
    filename = sprintf('%s_n100_k4_sample_allocation_%s.mat', network_type, scheme);

    % Load data
    data = load(filename);
    adj_field  = sprintf('adjacency_matrix_%s_k4_sample_allocation_%s', network_type, scheme);
    t_field    = sprintf('t_array_%s_k4_sample_allocation_%s', network_type, scheme);
    rhoc_field = sprintf('rhoc_array_%s_k4_sample_allocation_%s', network_type, scheme);
    rhod_field = sprintf('rhod_array_%s_k4_sample_allocation_%s', network_type, scheme);
    G = double(full(data.(adj_field)));
    s = double(data.(t_field)');
    rhoc = data.(rhoc_field);
    rhod = data.(rhod_field);

    % Compute sharer and withholder performance
    [sim_s_perf, sim_w_perf] = average_error_parallel(G, s, p, T);

    n_R = length(rhoc);
    collective_simu = zeros(1, n_R);
    for i = 1:n_R
        collective_simu(i) = sim_s_perf * rhoc(i) / (rhoc(i) + rhod(i)) + ...
                              sim_w_perf * rhod(i) / (rhoc(i) + rhod(i));
    end

    fprintf('--- %s, %s ---\n', network_type, scheme);
    disp(collective_simu);

    % Save results
    collective_field = sprintf('collective_simu_%s_k4_sample_allocation_%s', network_type, scheme);
    sim_s_perf_field  = sprintf('sim_s_perf_%s_k4_sample_allocation_%s', network_type, scheme);
    sim_w_perf_field  = sprintf('sim_w_perf_%s_k4_sample_allocation_%s', network_type, scheme);
    S = struct();
    S.(collective_field) = collective_simu;
    S.(sim_s_perf_field) = sim_s_perf;
    S.(sim_w_perf_field) = sim_w_perf;
    save(filename, '-struct', 'S', '-append');
end
