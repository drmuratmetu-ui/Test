function [R, r] = sum_rate(H, V, sigma2)
% SUM_RATE  Downlink sum rate (bit/s/Hz) with linear precoding.
%   H : K x M channel (row k = h_k^H),  V : M x K precoder,  sigma2 : noise power
%   R : sum rate,  r : K x 1 per-user rates
G    = abs(H * V).^2;                 % G(k,j) = |h_k^H v_j|^2
sig  = diag(G);
intf = sum(G, 2) - sig;
r    = log2(1 + sig ./ (intf + sigma2));
R    = sum(r);
end
