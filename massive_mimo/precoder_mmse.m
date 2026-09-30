function V = precoder_mmse(H, P, sigma2)
% PRECODER_MMSE  MMSE (regularized zero-forcing) precoding
%   V = H^H (H H^H + (K*sigma2/P) I)^-1, scaled to total power P.
K = size(H, 1);
V = H' / (H * H' + (K * sigma2 / P) * eye(K));
V = V * sqrt(P) / norm(V, 'fro');
end
