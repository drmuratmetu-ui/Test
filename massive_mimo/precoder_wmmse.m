function [V, hist] = precoder_wmmse(H, P, sigma2, V0, maxIter, tol)
% PRECODER_WMMSE  Weighted-MMSE sum-rate maximization (Shi, Razaviyayn, Luo,
%   IEEE Trans. Signal Process., 2011) for the multi-user MISO downlink.
%   Alternates between (1) MMSE receivers u_k, (2) MSE weights w_k = 1/e_k and
%   (3) the transmit precoder V under the power constraint ||V||_F^2 <= P.
%   The sum rate is non-decreasing across iterations, so initializing with
%   MMSE precoding gives WMMSE >= MMSE.
%
%   hist : sum rate after each iteration (hist(1) is the initial point)
if nargin < 5, maxIter = 100;  end
if nargin < 6, tol     = 1e-5; end
K = size(H, 1);
G = H * H';                            % K x K Gram matrix
V = V0;
hist = zeros(maxIter + 1, 1);
hist(1) = sum_rate(H, V, sigma2);

for it = 1:maxIter
    % (1) MMSE receiver: s_hat_k = conj(u_k) * y_k
    HV  = H * V;
    sig = diag(HV);
    den = sum(abs(HV).^2, 2) + sigma2;
    u   = sig ./ den;
    % (2) MSE weights
    e = 1 - abs(sig).^2 ./ den;
    w = 1 ./ e;
    % (3) Precoder: V = (H^H D H + mu I)^-1 H^H diag(w.*u), D = diag(w|u|^2).
    %     Push-through identity keeps it K x K: V = H^H (D G + mu I)^-1 diag(w.*u)
    DG = diag(w .* abs(u).^2) * G;
    C  = diag(w .* u);
    X  = DG \ C;
    if txpower(X, G) > P               % bisection on the Lagrange multiplier mu
        lo = 0; hi = 1;
        while txpower((DG + hi * eye(K)) \ C, G) > P, hi = 2 * hi; end
        for b = 1:60
            mid = (lo + hi) / 2;
            if txpower((DG + mid * eye(K)) \ C, G) > P, lo = mid; else, hi = mid; end
            if hi - lo < 1e-12 * hi, break; end
        end
        X = (DG + hi * eye(K)) \ C;
    end
    V = H' * X;

    hist(it + 1) = sum_rate(H, V, sigma2);
    if abs(hist(it + 1) - hist(it)) < tol * abs(hist(it))
        hist = hist(1:it + 1);
        return;
    end
end
end

function p = txpower(X, G)
% ||H^H X||_F^2 = tr(X^H G X)
p = real(trace(X' * G * X));
end
