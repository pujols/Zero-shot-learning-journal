function Sim = Compute_Sim(sig_Y, sig_R, Sim_scale, Sim_type)

if (strcmp(Sim_type, 'RBF_norm'))
    % disp('RBF_norm');
    dist = pdist2_fast(sig_Y, sig_R);
    Sim = exp(-dist .^ 2 * Sim_scale);
    Sim = bsxfun(@rdivide, Sim, sum(Sim, 2));
elseif (strcmp(Sim_type, 'RBF'))
    % disp('RBF');
    dist = pdist2_fast(sig_Y, sig_R);
    Sim = exp(-dist .^ 2 * Sim_scale);
elseif (strcmp(Sim_type, 'inner'))
    Sim = sig_Y * sig_R';
elseif (strcmp(Sim_type, 'inner_L2'))
    Sim = sig_Y * sig_R';
    Sim = bsxfun(@rdivide, Sim, sqrt(sum(Sim .^ 2, 2)));
elseif (strcmp(Sim_type, 'inner_L1'))
    Sim = sig_Y * sig_R';
    Sim = bsxfun(@rdivide, Sim, sum(abs(Sim), 2));
elseif (strcmp(Sim_type, 'inner_positive'))
    Sim = sig_Y * sig_R';
    Sim(Sim < 0) = 0;
elseif (strcmp(Sim_type, 'inner_L1_positive'))
    Sim = sig_Y * sig_R';
    Sim(Sim < 0) = 0;
    Sim = bsxfun(@rdivide, Sim, sqrt(sum(Sim .^ 2, 2)));
elseif (strcmp(Sim_type, 'inner_L2_positive'))
    Sim = sig_Y * sig_R';
    Sim(Sim < 0) = 0;
    Sim = bsxfun(@rdivide, Sim, sum(abs(Sim), 2));
end
Sim(isnan(Sim)) = 0; Sim(isinf(Sim)) = 0;
end