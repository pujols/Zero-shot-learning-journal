function [score_S, score_U] = test_V_GZSL(V, Sim_S, Sim_U, X)

W_S = construct_W(V, Sim_S);
W_U = construct_W(V, Sim_U);
score_S = X * W_S';
score_U = X * W_U';
end
