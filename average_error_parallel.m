function [full_sharing, full_withholding] = average_error_parallel(G, s, p, T)
N = size(G, 1);
K = length(p);

% --- Precomputation ---
G_augmented = G + eye(N);
s_full = G_augmented * s;

% --- Parallel simulation ---
errors_full = zeros(T, 1);
errors_non = zeros(T, 1);

parfor t = 1:T
    % Generate samples
    original_samples = zeros(N, K);
    for i = 1:N
        original_samples(i, :) = mnrnd(s(i), p);
    end
    
    % Full sharing error
    aggregated = G_augmented * original_samples;
    estimates_full = aggregated ./ s_full;
    diff_full = estimates_full - p;
    errors_full(t) = sum(sum(diff_full.^2, 2));
    
    % Full withholding error
    estimates_non = original_samples ./ s;
    diff_non = estimates_non - p;
    errors_non(t) = sum(sum(diff_non.^2, 2));
end

% --- Average errors ---
full_sharing = sum(errors_full) / (N * T);
full_withholding = sum(errors_non) / (N * T);

end