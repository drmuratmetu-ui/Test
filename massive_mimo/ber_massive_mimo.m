% BER_MASSIVE_MIMO  Uncoded QPSK bit error rate vs SNR for MRC, MMSE and WMMSE
%   precoding in the massive MIMO downlink (same system as main_massive_mimo.m).
%
%   Each user knows its effective gain g_k = h_k^H v_k (e.g. from downlink
%   pilots), equalizes y_k / g_k and makes a hard Gray-coded QPSK decision.
%   Two BER estimates are computed:
%     - Monte Carlo: random QPSK symbols sent through the actual channels,
%       bit errors counted (markers; shown where >= minErr errors were seen).
%     - Semi-analytic: BER_k = Q(sqrt(SINR_k)), treating the residual
%       interference as Gaussian (lines; reaches BERs Monte Carlo cannot).
%   Both are averaged over all K users, so users that WMMSE switches off
%   (zero power, BER = 1/2) are counted. WMMSE maximizes sum rate, not BER.
%
%   Saves ber_vs_snr.png and ber_vs_snr.mat. Runs in MATLAB and GNU Octave.

clear; close all; clc;
rng(3);
if exist('OCTAVE_VERSION', 'builtin'), warning('off', 'Octave:gnuplot-graphics'); end

%% Parameters
K       = 16;                   % users
M       = 64;                   % BS antennas
SNRdB   = -10:2.5:20;
nReal   = 200;                  % channel realizations per SNR point
nSym    = 500;                  % QPSK symbol vectors per channel realization
minErr  = 20;                   % Monte Carlo points need at least this many errors
sigma2  = 1;
maxIter = 100;  tol = 1e-5;     % WMMSE stopping rule

names = {'MRC', 'MMSE', 'WMMSE'};
chan  = @(K, M) (randn(K, M) + 1i * randn(K, M)) / sqrt(2);
Qf    = @(x) 0.5 * erfc(x / sqrt(2));

nS       = numel(SNRdB);
ber_an   = zeros(nS, 3);        % semi-analytic BER
err_mc   = zeros(nS, 3);        % Monte Carlo bit errors
nOff     = zeros(nS, 1);        % WMMSE users switched off (average per channel)
nBits    = nReal * nSym * K * 2;

tic;
for s = 1:nS
    P = 10^(SNRdB(s) / 10);
    for n = 1:nReal
        H = chan(K, M);
        V = cell(1, 3);
        V{1} = precoder_mrt(H, P);
        V{2} = precoder_mmse(H, P, sigma2);
        V{3} = precoder_wmmse(H, P, sigma2, V{2}, maxIter, tol);
        nOff(s) = nOff(s) + sum(sum(abs(V{3}).^2, 1) < 1e-9 * P) / nReal;

        % Same symbols and noise for every precoder (common random numbers)
        b1 = rand(K, nSym) > 0.5;
        b2 = rand(K, nSym) > 0.5;
        S  = ((2 * b1 - 1) + 1i * (2 * b2 - 1)) / sqrt(2);
        N  = sqrt(sigma2 / 2) * (randn(K, nSym) + 1i * randn(K, nSym));

        for a = 1:3
            HV  = H * V{a};
            g   = diag(HV);                         % effective gains
            % SINR -> semi-analytic BER
            pw  = abs(HV).^2;
            sinr = abs(g).^2 ./ (sum(pw, 2) - abs(g).^2 + sigma2);
            ber_an(s, a) = ber_an(s, a) + mean(Qf(sqrt(sinr))) / nReal;
            % Monte Carlo: equalize and detect (g = 0 -> random guess)
            Y  = HV * S + N;
            Z  = Y ./ g;
            Z(~isfinite(Z)) = 0;
            e  = xor(real(Z) > 0, b1) + xor(imag(Z) > 0, b2);
            err_mc(s, a) = err_mc(s, a) + sum(e(:));
        end
    end
    fprintf('SNR %5.1f dB done (%.0f s)\n', SNRdB(s), toc);
end
ber_mc = err_mc / nBits;

%% Print table
fprintf('\nUncoded QPSK BER vs SNR  (M = %d, K = %d, %d channels x %d symbols)\n', M, K, nReal, nSym);
fprintf('Semi-analytic Q(sqrt(SINR)) | Monte Carlo (- : fewer than %d errors)\n', minErr);
fprintf('%8s | %10s %10s %10s | %10s %10s %10s | %s\n', 'SNR(dB)', names{:}, names{:}, 'WMMSE users off');
for s = 1:nS
    mc = cell(1, 3);
    for a = 1:3
        if err_mc(s, a) >= minErr, mc{a} = sprintf('%10.2e', ber_mc(s, a));
        else,                       mc{a} = sprintf('%10s', '-'); end
    end
    fprintf('%8.1f | %10.2e %10.2e %10.2e | %s %s %s | %.2f\n', SNRdB(s), ber_an(s, :), mc{:}, nOff(s));
end

%% Plot
col = [0.165 0.471 0.839;  0.922 0.408 0.204;  0.106 0.686 0.478];  % blue, orange, aqua
mk  = {'o', 's', '^'};
fig = figure('visible', 'off', 'position', [0 0 760 560]);
h = zeros(1, 3);
for a = 1:3
    h(a) = semilogy(SNRdB, max(ber_an(:, a), 1e-12), '-', 'color', col(a, :), 'linewidth', 2);
    hold on;
    ok = err_mc(:, a) >= minErr;
    semilogy(SNRdB(ok), ber_mc(ok, a), mk{a}, 'color', col(a, :), ...
             'markersize', 8, 'markerfacecolor', col(a, :));
end
grid on; box off;
ylim([1e-6 1]); xlim([SNRdB(1) SNRdB(end)]);
xlabel('SNR (dB)'); ylabel('Bit error rate');
title(sprintf('Uncoded QPSK BER  (M = %d, K = %d)', M, K));
legend(h, names, 'location', 'southwest');
text(SNRdB(end) - 0.5, 4e-1, {'lines: Q(\surd SINR)', 'markers: Monte Carlo'}, ...
     'horizontalalignment', 'right', 'fontsize', 9);

print(fig, 'ber_vs_snr.png', '-dpng', '-r110');
save('-v7', 'ber_vs_snr.mat', 'SNRdB', 'ber_an', 'ber_mc', 'err_mc', 'nBits', 'nOff', 'M', 'K', 'nReal', 'nSym');
fprintf('\nSaved ber_vs_snr.png and ber_vs_snr.mat\n');
