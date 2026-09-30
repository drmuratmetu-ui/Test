function V = precoder_mrt(H, P)
% PRECODER_MRT  Maximum-ratio (MRC/MRT) precoding, V = H^H scaled to total power P.
V = H';
V = V * sqrt(P) / norm(V, 'fro');
end
