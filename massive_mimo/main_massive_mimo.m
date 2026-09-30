% MAIN_MASSIVE_MIMO  Downlink massive MIMO: MRC (MRT) vs MMSE vs WMMSE precoding.
%
%   System: one base station with M antennas serves K single-antenna users.
%   Channel: i.i.d. Rayleigh fading, h_k ~ CN(0, I_M), perfect CSI at the BS.
%   Power:   total transmit power P, noise power sigma2 = 1, so SNR = P.
%   Metric:  ergodic sum spectral efficiency (bit/s/Hz), averaged over channels.
%
%   Produces (1) sum rate vs SNR, (2) sum rate vs M, (3) WMMSE convergence,
%   saved to massive_mimo_results.png, and prints the numbers as tables.
%   Runs in MATLAB and GNU Octave.

clear; close all; clc;
rng(1);

%% Parameters
K        = 16;                  % users
M_fixed  = 64;                  % BS antennas for the SNR sweep
SNRdB    = -10:5:30;            % SNR sweep
M_list   = [16 24 32 48 64 96 128];
SNR_M_dB = 10;                  % SNR for the antenna sweep
nReal    = 200;                 % channel realizations per point
sigma2   = 1;
maxIter  = 100;  tol = 1e-5;    % WMMSE stopping rule

names = {'MRC', 'MMSE', 'WMMSE'};
chan  = @(K, M) (randn(K, M) + 1i * randn(K, M)) / sqrt(2);

%% Experiment 1: sum rate vs SNR
R_snr = zeros(numel(SNRdB), 3);
tic;
for s = 1:numel(SNRdB)
    P = 10^(SNRdB(s) / 10);
    for n = 1:nReal
        H  = chan(K, M_fixed);
        V1 = precoder_mrt(H, P);
        V2 = precoder_mmse(H, P, sigma2);
        V3 = precoder_wmmse(H, P, sigma2, V2, maxIter, tol);
        R_snr(s, :) = R_snr(s, :) + [sum_rate(H, V1, sigma2), ...
                                     sum_rate(H, V2, sigma2), ...
                                     sum_rate(H, V3, sigma2)] / nReal;
    end
end
fprintf('Experiment 1 done in %.1f s\n', toc);

%% Experiment 2: sum rate vs number of BS antennas
P = 10^(SNR_M_dB / 10);
R_M = zeros(numel(M_list), 3);
tic;
for m = 1:numel(M_list)
    for n = 1:nReal
        H  = chan(K, M_list(m));
        V1 = precoder_mrt(H, P);
        V2 = precoder_mmse(H, P, sigma2);
        V3 = precoder_wmmse(H, P, sigma2, V2, maxIter, tol);
        R_M(m, :) = R_M(m, :) + [sum_rate(H, V1, sigma2), ...
                                 sum_rate(H, V2, sigma2), ...
                                 sum_rate(H, V3, sigma2)] / nReal;
    end
end
fprintf('Experiment 2 done in %.1f s\n', toc);

%% Experiment 3: WMMSE convergence on one channel (M = K = 16, where it matters most)
P = 10^(SNR_M_dB / 10);
H = chan(K, K);
R_mrc  = sum_rate(H, precoder_mrt(H, P), sigma2);
V2     = precoder_mmse(H, P, sigma2);
R_mmse = sum_rate(H, V2, sigma2);
[~, histMMSEinit] = precoder_wmmse(H, P, sigma2, V2, 200, 1e-7);
[~, histMRCinit]  = precoder_wmmse(H, P, sigma2, precoder_mrt(H, P), 200, 1e-7);

%% Print tables
fprintf('\nSum rate vs SNR  (M = %d, K = %d, %d realizations) [bit/s/Hz]\n', M_fixed, K, nReal);
fprintf('%8s %8s %8s %8s %10s\n', 'SNR(dB)', names{:}, 'WMMSE-MMSE');
for s = 1:numel(SNRdB)
    fprintf('%8d %8.2f %8.2f %8.2f %10.2f\n', SNRdB(s), R_snr(s, :), R_snr(s, 3) - R_snr(s, 2));
end
fprintf('\nSum rate vs M  (K = %d, SNR = %d dB) [bit/s/Hz]\n', K, SNR_M_dB);
fprintf('%8s %8s %8s %8s %10s\n', 'M', names{:}, 'WMMSE-MMSE');
for m = 1:numel(M_list)
    fprintf('%8d %8.2f %8.2f %8.2f %10.2f\n', M_list(m), R_M(m, :), R_M(m, 3) - R_M(m, 2));
end
fprintf('\nWMMSE convergence (M = K = %d, SNR = %d dB): MRC %.2f, MMSE %.2f, ', K, SNR_M_dB, R_mrc, R_mmse);
fprintf('WMMSE %.2f (from MMSE, %d it), %.2f (from MRC, %d it)\n', ...
        histMMSEinit(end), numel(histMMSEinit) - 1, histMRCinit(end), numel(histMRCinit) - 1);

%% Plot
col = [0.165 0.471 0.839;  0.922 0.408 0.204;  0.106 0.686 0.478];  % blue, orange, aqua
mk  = {'o', 's', '^'};
fig = figure('visible', 'off', 'position', [0 0 1500 460]);

subplot(1, 3, 1); hold on;
for a = 1:3
    plot(SNRdB, R_snr(:, a), ['-' mk{a}], 'color', col(a, :), 'linewidth', 2, ...
         'markersize', 7, 'markerfacecolor', col(a, :));
end
grid on; box off; xlabel('SNR (dB)'); ylabel('Sum rate (bit/s/Hz)');
title(sprintf('Sum rate vs SNR  (M = %d, K = %d)', M_fixed, K));
legend(names, 'location', 'northwest');

subplot(1, 3, 2); hold on;
for a = 1:3
    plot(M_list, R_M(:, a), ['-' mk{a}], 'color', col(a, :), 'linewidth', 2, ...
         'markersize', 7, 'markerfacecolor', col(a, :));
end
grid on; box off; xlabel('BS antennas M'); ylabel('Sum rate (bit/s/Hz)');
title(sprintf('Sum rate vs M  (K = %d, SNR = %d dB)', K, SNR_M_dB));
legend(names, 'location', 'southeast');

subplot(1, 3, 3); hold on;
plot(0:numel(histMRCinit) - 1,  histMRCinit,  '-', 'color', col(1, :), 'linewidth', 2);
plot(0:numel(histMMSEinit) - 1, histMMSEinit, '-', 'color', col(2, :), 'linewidth', 2);
plot([0 numel(histMRCinit) - 1], [R_mmse R_mmse], '--', 'color', [0.45 0.45 0.45], 'linewidth', 1.5);
grid on; box off; xlabel('WMMSE iteration'); ylabel('Sum rate (bit/s/Hz)');
title(sprintf('WMMSE convergence  (M = K = %d, SNR = %d dB)', K, SNR_M_dB));
legend({'WMMSE, MRC init', 'WMMSE, MMSE init', 'MMSE (no iteration)'}, 'location', 'southeast');

print(fig, 'massive_mimo_results.png', '-dpng', '-r110');
save('-v7', 'massive_mimo_results.mat', 'SNRdB', 'R_snr', 'M_list', 'R_M', 'histMRCinit', 'histMMSEinit', 'K', 'M_fixed', 'SNR_M_dB', 'nReal');
fprintf('\nSaved massive_mimo_results.png and massive_mimo_results.mat\n');
