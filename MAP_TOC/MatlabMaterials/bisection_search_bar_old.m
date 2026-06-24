function lambda = bisection_search_bar(lambda_lb, lambda_ub, P, epsilon, g, Gamma, matrix_form)

lb = lambda_lb;
ub = lambda_ub;
lambda = lambda_ub;

max_iter = 50;   % safety cap
iter = 0;

prev_width = ub - lb;

while (ub - lb >= epsilon) && (iter < max_iter)

    lambda = (lb + ub)/2;

    val = f_bar(lambda, g, Gamma, matrix_form);

    % safety check (VERY important)
    if ~isfinite(val)
        val = P + 1;
    end

    if val <= P
        ub = lambda;
    else
        lb = lambda;
    end

    % ---- stagnation detection ----
    new_width = ub - lb;

    if abs(new_width - prev_width) < 1e-12
        break;   % interval no longer shrinking
    end

    prev_width = new_width;
    iter = iter + 1;
end

end




% 
% 
% 
% function lambda = bisection_search_bar(lambda_lb, lambda_ub, P, epsilon, g, Gamma, matrix_form)
% %   Detailed explanation goes here
% 
% lb = lambda_lb;
% ub = lambda_ub;
% lambda = lambda_ub;
% iii=1;
% while 1
%     % ff(iii) = ub - lb ;
%     lambda = (lb + ub)/2;
%     if f_bar(lambda, g, Gamma, matrix_form) <= P
%         ub = lambda;
%     else
%         lb = lambda;
%     end
%     ff(iii) = ub - lb;
%     if iii>20
%         if ff(iii)==mean(ff(iii-20:iii))
%             break
%         end
%     end
%     iii=1+iii;
% end
% 
% end