
function [theta_out, iteration,all_thetas] = algoEM(vect_x, theta_param_0)

epsilon = 0.1;
max_iter = 1000;

all_thetas = [];
theta_cell = num2cell(theta_param_0);
[pi_1, pi_2, m_1, sig_1, m_2, sig_2] = theta_cell{:};

for iteration = 1:max_iter
    

    t_1 = pi_1 * exp((-1/2) * ((vect_x - m_1) / abs(sig_1)).^2) / (abs(sig_1) * sqrt(2*pi));
    t_2 = pi_2 * exp((-1/2) * ((vect_x - m_2) / abs(sig_2)).^2) / (abs(sig_2) * sqrt(2*pi));
    t_sum = t_2 + t_1;
    t_1 = t_1 ./ t_sum;
    t_2 = t_2 ./ t_sum;
    
    prior_theta = [pi_1, pi_2, m_1, sig_1, m_2, sig_2];
    
    pi_1 = sum(t_1) / length(vect_x);
    pi_2 = sum(t_2) / length(vect_x);
    
    m_1 = sum(t_1 .* vect_x) / sum(t_1);
    m_2 = sum(t_2 .* vect_x) / sum(t_2);
    
    sig_1 = sqrt(sum(t_1 .* (vect_x - m_1).^2) / sum(t_1));
    sig_2 = sqrt(sum(t_2 .* (vect_x - m_2).^2) / sum(t_2));
    
    theta_out = [pi_1, pi_2, m_1, sig_1, m_2, sig_2];
    
    all_thetas = [all_thetas, {theta_out}];
    if norm(theta_out - prior_theta) < epsilon
       %Pass
    end
end

if iteration == max_iter
    disp("Convergence not reached within maximum iterations.");
end

end
