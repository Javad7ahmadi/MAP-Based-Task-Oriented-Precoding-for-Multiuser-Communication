clc;clear;close all;
mssag = 'Diff_T';
disp('please choose the precoder, the number of viewers and feature length')
DetectorType = 1; % 1: exact MAP detector, 2: approximate map detector:

PrecoderDesign   = 1; % PrecoderDesign = input('choose... 1:MAP: , 2:MCR2 , 3:LMMSE, 0:Random .... ');

rho = 0.0; % channel correlation



Ite = 10;
ITERATIONS = 1000;
T_vector =  [1 2 3 4 5 8]; %  power dB
Eta_vec = 10;  % \eta: step size
Acc_Ours_2 = zeros(size(length(Eta_vec),length(T_vector)));
for iter = 1:ITERATIONS




    load('saved_MAT.mat');



    p=[0.10000000	0.10000000	0.10000000	0.10000000	0.10000000	0.10000000	0.10000000	0.10000000	0.10000000	0.10000000];

    epochNum = 20;

    CLS_mean     = double(squeeze(CLS_mean(epochNum,:,:)));
    CLS_cov      = double(squeeze(CLS_cov(epochNum,:,:,:)));
    test_label   = double(test_label2(epochNum,:));%test_labels(1:1000);
    test_feature = double(squeeze(test_feature_cplx2(epochNum,:,:)).');
    I_1 = 100;
    % B = 10*1e3;  % 10 kHz
    K = double(K);  % num. of devices
    N_r = 4;
    N_t_k = 2;
    N_t = N_t_k * K;
    P_dBm = 30;  % dBm
    P = 1e-3*10.^(P_dBm./10);  % W
    P_k = P/ K;

    Dataset = size(test_feature, 2);
    C = double(squeeze(C_xx(epochNum,:,:)));
    L = length(class_priors);  % num. of classes
    D = 2*size(test_feature, 1); % the muliply 2 is for real and imag
    delta_0 = 1;  % normalize path loss into the noise power
    Acc_Ourss = zeros(size(length(Eta_vec),length(T_vector)));
    for varInx1 = 1:length(Eta_vec)
        eta = Eta_vec(varInx1);  % num. of time slots
        mssges{varInx1} = ['$\eta = $' num2str(eta)];
        for varInx2 = 1:length(T_vector)
            T = T_vector(varInx2);  % num. of time slots
            % P_dBm = Var_Vec2(varInx2);  % dBm
            % P = 1e-3*10.^(P_dBm./10);  % W
            % P_k = P/ K;


            % P_k_dBm = 30;  % dBm
            % P_k = 1e-3*10.^(P_k_dBm./10);  % W
            % P = P_k * K;

            % channel generation
            % kappa = 1;  % Rician factor
            H = [];
            % Ht = [];
            H_k_all = zeros(T*N_r, T*N_t_k, K);
            % H_k_t_all = zeros(N_r, N_t_k, K);
            for k = 1:K
                if rho~=0
                    R_r = rho .^ abs((0:N_r-1)' - (0:N_r-1));

                    H_iid = (randn(N_r, N_t_k) + 1j*randn(N_r, N_t_k)) / sqrt(2);

                    H_k_t = sqrtm(R_r) * H_iid;
                elseif rho==0
                    H_k_t = (randn(N_r, N_t_k) + 1j*randn(N_r, N_t_k)) / sqrt(2);
                end
                % H_k_t_all(:, :, k) = H_k_t;
                %             svd(H_k_t)
                H_k = kron(eye(T), H_k_t);
                H_k_all(:, :, k) = H_k;
                H = [H, H_k];
                % Ht = [Ht, H_k_t];
            end
            disp(['Iter ' num2str(iter) ', var  = ' num2str(varInx1)])
            if PrecoderDesign==1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%% Our Precoder %%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                tic;
                V = Our_Encoder(H,P, CLS_mean, CLS_cov, p, delta_0, K, N_t_k,N_r, T,eta,Ite);
                Time_Mine(iter,varInx2) = toc;
                %%%%%%MAP receiver %%%%%%%
                [gamma_G(:,iter),Acc_Ourss(varInx1,varInx2)] = OurChannel_Receiver( ...
                    H,  CLS_cov, CLS_mean, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label,mssag,DetectorType);
            elseif PrecoderDesign==2
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%% MCR2 Precoder %%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                tic;

                [V, mcr2_all] = Training_MCR2_precoder( ...
                    H, C, CLS_cov, p, delta_0, P_k, K, N_t_k,N_r, T, D,P,L,H_k_all,I_1);
                Time_MCR2(iter,varInx2) = toc;
                %%%%%%MAP receiver %%%%%%%
                [gamma_G(:,iter),Acc_Ourss(varInx1,varInx2)] = OurChannel_Receiver( ...
                    H,  CLS_cov, CLS_mean, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label,mssag,DetectorType);



            elseif PrecoderDesign==3
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%% LMMSE Precoder%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                [V, mse_all] = LMMSE_precoder(H, C, delta_0, P_k, K, N_t_k,N_r, T, D,P,H_k_all,I_1);
                %%%%%%MAP receiver %%%%
                [gamma_G(:,iter),Acc_Ourss(varInx1,varInx2)] = OurChannel_Receiver( ...
                    H,  CLS_cov, CLS_mean,  delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label,mssag,DetectorType);
            elseif PrecoderDesign==0
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%% Random Precoder%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                N_T = K * N_t_k*T;
                V = randn(N_T, D/2) + 1j * randn(N_T, D/2);
                %%%%%%MAP receiver %%%%
                [gamma_G(:,iter),Acc_Ourss(varInx1,varInx2)] = OurChannel_Receiver( ...
                    H,  CLS_cov, CLS_mean, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label,mssag,DetectorType);
            end
        end
    end
    Acc_Ours_2 = Acc_Ours_2+Acc_Ourss;
    Acc_Ourss3 = Acc_Ours_2/iter;

    aa = 5;
    if round(iter/aa)==iter/aa
        close all;
        plot(T_vector, 1-Acc_Ourss3', 'LineWidth', 2); hold on
        ylabel('MAP error probability', 'Interpreter', 'latex')

        xlabel('T', 'Interpreter', 'latex')

        % legend(Mess, 'Interpreter', 'latex', 'Location', 'best')
        % legend(mssges, 'Interpreter', 'latex', 'Location', 'best')
        errorMat2 = 1-Acc_Ourss3;
        grid on
        box on
    end
    % xlim([1-.1,11])
    % drawnow
    % figure(823);hold all;plot(gamma_G(:,iter))
    % lowerbound(iter) = P*T/K;
    % goodestimate(iter) =P/K;
    % disp(estimate1) aa
end















function [V, mcr2_all] = Training_MCR2_precoder( ...
    H, C, CLS_cov, p, delta_0, P_k, K, N_t_k,N_r, T, D,P,L,H_k_all,I_1)
epsilons = 0.01;

alpha = T*N_r / epsilons^2;
C_sr = sqrtm(C);

% initialization of V
V_init_k = rand(T*N_t_k, D/(2*K)) + 1j*rand(T*N_t_k, D/(2*K));
V_init = kron(eye(K), V_init_k);
V_init = sqrt(P) * V_init ./ sqrt(trace(V_init*C*V_init'));
V = V_init;

Ite = 10;
mcr2_obj_last = 500;
mcr2_all = zeros(1, Ite);
for ii = 1:Ite
    ii;
    % U update
    U = (H*V*C*V'*H'+(1/alpha + delta_0^2)*eye(N_r*T))\(H*V*C_sr);

    % W update
    E_0 = eye(D/2)-U'*H*V*C_sr;
    E_0 = (E_0*E_0') + (1/alpha + delta_0^2)*(U'*U);
    W_0 = inv(E_0);
    W_j_all = zeros(N_r*T, N_r*T, L);
    for j = 1:L
        C_j = CLS_cov(:, :, j);
        W_j_all(:, :, j) = inv((1+alpha*delta_0^2)*eye(N_r*T) + alpha*H*V*C_j*V'*H');
    end

    % V update
    for k = 1:K
        H_k = H_k_all(:, :, k);
        C_sr_k = C_sr(((k-1)*D/(2*K)+1):k*D/(2*K), :);
        C_k = C(((k-1)*D/(2*K)+1):k*D/(2*K), :);
        C_kk = C(((k-1)*D/(2*K)+1):k*D/(2*K), ((k-1)*D/(2*K)+1):k*D/(2*K));
        V_k = V(((k-1)*T*N_t_k+1):k*T*N_t_k, ...
            ((k-1)*D/(2*K)+1):k*D/(2*K));

        sum_term1_T_k = H*V*C_k' - H_k*V_k*C_kk;
        sum_term2_T_k = 0;  % zeros(2*T*N_t_k, D/K);
        sum_term_M_k = 0;
        %                 I_kron_H_bar_k = kron(eye(D/(2*K)), H_k);
        for j = 1:L
            W_j = W_j_all(:, :, j);

            C_j_k = CLS_cov(((k-1)*D/(2*K)+1):k*D/(2*K), :, j);
            C_j_kk = CLS_cov(((k-1)*D/(2*K)+1):k*D/(2*K), ((k-1)*D/(2*K)+1):k*D/(2*K), j);
            sum_term2_j_T_k = H*V*C_j_k' - H_k*V_k*C_j_kk;
            sum_term2_T_k = sum_term2_T_k + ...
                alpha*p(j)*H_k'*W_j*sum_term2_j_T_k;

            %                     sum_term_M_k = sum_term_M_k + ...
            %                         alpha*p(j)*I_kron_H_bar_k'*kron(C_j_kk, W_j)*I_kron_H_bar_k;
            sum_term_M_k = sum_term_M_k + ...
                alpha*p(j)*kron(C_j_kk.', H_k'*W_j*H_k);
        end
        T_k = H_k'*U*W_0*C_sr_k' - ...
            H_k'*U*W_0*U'*sum_term1_T_k - sum_term2_T_k;
        %                 I_kron_U_H_bar_k = kron(eye(D/K), U'*H_k);
        M_k = kron(C_kk.', H_k'*U*W_0*U'*H_k) + ...
            sum_term_M_k;

        [Q_kk, D_kk] = svd(C_kk.');
        Q_kk_kron_I = kron(Q_kk, eye(N_t_k*T));
        D_kk_kron_I = kron(D_kk, eye(N_t_k*T));
        D_kk_kron_I_msr = diag( 1./sqrt(diag(D_kk_kron_I)) );
        C_kk_kron_I_msr = Q_kk_kron_I * D_kk_kron_I_msr * Q_kk_kron_I';

        % t_k = C_kk_kron_I_msr*vec(T_k);
        t_k = C_kk_kron_I_msr * T_k(:);
        M_k = C_kk_kron_I_msr*M_k*C_kk_kron_I_msr;


        [U_M_k, Gamma_k] = svd(M_k);
        g_k = U_M_k' * t_k;

        lambda_lb = 0;
        lambda_ub = sqrt(real(g_k'*g_k)/(P_k*T));

        fcn_bar_0 = f_bar(0, g_k, Gamma_k, false);
        epsilon_bi_search = 1e-6;
        if fcn_bar_0 <= P_k
            lambda = 0;
        else
            lambda = bisection_search_bar(lambda_lb, lambda_ub, T*P_k, epsilon_bi_search, g_k, Gamma_k, false,I_1);
        end

        if lambda < 1e-20
            v_k = pinv( M_k + lambda*eye(N_t_k*T*D/(2*K)) ) * t_k;
        else
            v_k = ( M_k + lambda*eye(N_t_k*T*D/(2*K)) ) \ t_k;
        end

        v_k = C_kk_kron_I_msr * v_k;

        % retrive V_k from v_k
        V_k = zeros(T*N_t_k, D/(2*K));
        for i = 1:D/(2*K)
            V_k(:, i) = v_k(T*N_t_k*(i-1)+1: T*N_t_k*i);
        end
        V((T*N_t_k*(k-1)+1):T*N_t_k*k, ((k-1)*D/(2*K)+1):k*D/(2*K)) = V_k;
    end
    [mcr2, ~, ~] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p);
    mcr2_all(ii) = mcr2;

    % terminate criterion
    if abs((mcr2 - mcr2_obj_last)/mcr2_obj_last) < 1e-5
        break;
    end
    mcr2_obj_last = mcr2;
end
% figure;
% plot(mcr2_all(1:ii), 'r-', LineWidth=1.6);
% [mcr2, ~, ~] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p);
end




function acc_mcr2 = Channel_Receiver( ...
    H,  CLS_cov, CLS_mean, CLS_rlt, p, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label)
epsilon = 1e-6; % trick
acc_mcr2 = 0;
for j = 1:Dataset
    z_cplx = test_feature(:, j);
    % add noise
    n_cplx = delta_0 * sqrt(1/2) * (randn(N_r*T, 1) + 1j*randn(N_r*T, 1));
    C_n_cplx = delta_0^2 * eye(N_r*T);
    signal = V*z_cplx;
    Pow1(j) = norm(signal, 'fro')^2;
    DesiredPower = P_k*K*T;
    y_cplx =sqrt(DesiredPower/Pow1(j))* H*V*z_cplx + n_cplx;
    signal2 = sqrt(DesiredPower/Pow1(j))* V*z_cplx;
    Pow2(j) = norm(signal2, 'fro')^2;

    %% Recover signals from complex to real
    z_real = [real(z_cplx); imag(z_cplx)];
    y_real = [real(y_cplx); imag(y_cplx)];
    HV_real = [real(H*V), -imag(H*V);...
        imag(H*V), real(H*V)];
    C_n_real = 1/2 * [real(C_n_cplx), zeros(N_r*T, N_r*T);
        zeros(N_r*T, N_r*T), real(C_n_cplx)];

    %%
    f = zeros(1, L);
    log_f = zeros(1, L); % numerical issue
    weighted_sum_element = zeros(1, L);
    for i = 1:L
        mu_i_cplx = CLS_mean(:, i);
        C_i = CLS_cov(:, :, i);
        R_i = CLS_rlt(:, :, i);

        mu_i = [real(mu_i_cplx); imag(mu_i_cplx)];
        Cov_i = 1/2 * [real(C_i + R_i), imag(-C_i + R_i);...
            imag(C_i + R_i), real(C_i - R_i)];

        [f(i), log_f(i)] = gaussian_pdf(y_real, HV_real*mu_i,...
            HV_real*Cov_i*HV_real' + C_n_real, epsilon);
        weighted_sum_element(i) = p(i) * f(i);
    end
    weighted_sum = sum(weighted_sum_element);

    % GM receiver
    %             [~, pos] = max(log_f);
    [~, pos] = max(weighted_sum_element);
    if (pos-1)==test_label(j)
        acc_mcr2 = acc_mcr2 +1;
    end


end
acc_mcr2 = acc_mcr2/Dataset;
end







function [V,mse_all] = LMMSE_precoder(H, C,  delta_0, P_k, K, N_t_k,N_r, T, D,P,H_k_all,I_1)
% epsilons
% alpha = T*N_r / epsilons^2;

% initialization of V
V_init_k = rand(T*N_t_k, D/(2*K)) + 1j*rand(T*N_t_k, D/(2*K));
V_init = kron(eye(K), V_init_k);
V_init = sqrt(P) * V_init ./ sqrt(trace(V_init*C*V_init'));
V = V_init;

Ite = 20;
mse_obj_last = 500;
mse_all = zeros(1, Ite);
for ii = 1:Ite
    ii;
    % R update
    R = (H*V*C*V'*H'+(delta_0^2)*eye(N_r*T))\(H*V*C);

    % V update
    for k = 1:K
        H_k = H_k_all(:, :, k);
        C_k = C(((k-1)*D/(2*K)+1):k*D/(2*K), :);
        C_kk = C(((k-1)*D/(2*K)+1):k*D/(2*K), ((k-1)*D/(2*K)+1):k*D/(2*K));
        V_k = V(((k-1)*T*N_t_k+1):k*T*N_t_k, ...
            ((k-1)*D/(2*K)+1):k*D/(2*K));

        sum_term_T_k = H*V*C_k' - H_k*V_k*C_kk;
        J_k = C_k - sum_term_T_k'*R;

        Q_k = H_k'*R*J_k'/C_kk;
        M_k = H_k'*R*R'*H_k;

        [U_M_k, Gamma_k] = svd(M_k);
        G_k = U_M_k'*Q_k*C_kk*Q_k'*U_M_k;

        lambda_lb = 0;
        lambda_ub = sqrt(real(sum(diag(G_k)))/(T*P_k));

        fcn_bar_0 = f_bar(0, diag(G_k), Gamma_k, true);
        epsilon_bi_search = 1e-6;
        if fcn_bar_0 <= T*P_k
            lambda = 0;
        else
            lambda = bisection_search_bar(lambda_lb, lambda_ub, T*P_k, epsilon_bi_search, diag(G_k), Gamma_k, true,I_1);
        end
        V_k = (M_k + lambda*eye(size(M_k)))\Q_k;

        V((T*N_t_k*(k-1)+1):T*N_t_k*k, ((k-1)*D/(2*K)+1):k*D/(2*K)) = V_k;
    end
    matrix_inv = inv(H*V*C*V'*H'+delta_0^2*eye(N_r*T));
    mse_obj = real( trace(C) - trace(C*V'*H'*matrix_inv*H*V*C) );
    mse_all(ii) = mse_obj;

    % terminate criterion
    if abs((mse_obj - mse_obj_last)/mse_obj_last) < 1e-5
        break;
    end
    mse_obj_last = mse_obj;
end
% figure;
% plot(mse_all(1:ii), 'r-', LineWidth=1.6)
% [mcr2, ~, ~] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p);
end



% function V = Our_Encoder(H, P, CLS_mean, CLS_cov, p, delta_0, K, N_t_k, N_r, T,eta,Ite)
%
% % ============================================================
% % Precoder optimization based on the new MAP approximation
% % ============================================================
%
% % tau =       % Q-function exponential approximation parameter
% tau = 0.7;      % Q-function exponential approximation parameter
%
% % Ite = 30;       % Number of iterations
% % eta = 0.1;      % Gradient-descent step size
%
% % ============================================================
% % Dimensions
% % ============================================================
%
% D = size(CLS_mean, 1);
% C = size(CLS_mean, 2);
%
% R = N_r * T;
% N = K * N_t_k;
%
% % Effective channel
% H_tilde = H;
%
% % Dimension of V:
% % V : (T*N) x D
%
% N_T = T * N;
%
% % ============================================================
% % Initialization of V
% % ============================================================
%
% V = randn(N_T, D) + 1j * randn(N_T, D);
%
% % ============================================================
% % Initial power projection
% % ============================================================
%
% Power = 0;
%
% for j = 1:C
%
%     mu_j = CLS_mean(:,j);
%     Sigma_j = CLS_cov(:,:,j);
%
%     Power = Power + p(j) * ( ...
%         real(trace(V * Sigma_j * V')) + ...
%         real(norm(V * mu_j)^2) ...
%         );
%
% end
%
%
% V = sqrt(P*T / Power) * V;
%
%
%
% % ============================================================
% % Optimization
% % ============================================================
%
% Error = zeros(Ite,1);
%
% for it = 1:Ite
%
%     % --------------------------------------------------------
%     % Step 1:
%     % Compute g_{j,m}(V) using CURRENT V
%     %
%     % These values are then FIXED during this GD iteration.
%     % --------------------------------------------------------
%
%     G_cov = zeros(C,R);
%
%     for j = 1:C
%
%         Sigma_j = CLS_cov(:,:,j);
%
%         for m = 1:R
%
%             h_m = H_tilde(m,:);
%
%             G_cov(j,m) = real( ...
%                 h_m * V * Sigma_j * V' * h_m' ...
%                 ) + delta_0;
%
%             % Add noise variance here if your sigma^2 is available.
%             % G_cov(j,m) = G_cov(j,m) + sigma2;
%
%         end
%
%     end
%
%
%     % --------------------------------------------------------
%     % Step 2:
%     % Calculate gradient
%     % --------------------------------------------------------
%
%     gradV = zeros(size(V));
%
%     total_error = 0;
%
%
%     for j = 1:C
%
%         for k = 1:C
%
%             if j == k
%                 continue;
%             end
%
%
%             % ------------------------------------------------
%             % Mean difference
%             % ------------------------------------------------
%
%             mu_j = CLS_mean(:,j);
%             mu_k = CLS_mean(:,k);
%
%             mu_jk = mu_j - mu_k;
%
%
%             % ------------------------------------------------
%             % A_jk
%             % ------------------------------------------------
%
%             A = 0;
%
%             for m = 1:R
%
%                 h_m = H_tilde(m,:);
%
%                 g_j = G_cov(j,m);
%                 g_k = G_cov(k,m);
%
%                 s = h_m * V * mu_jk;
%
%                 A = A + ...
%                     abs(s)^2 / g_k;
%
%             end
%
%
%             % ------------------------------------------------
%             % a''_jk
%             % ------------------------------------------------
%
%             a_jk = log(p(j)/p(k)) - R;
%
%             for m = 1:R
%
%                 g_j = G_cov(j,m);
%                 g_k = G_cov(k,m);
%
%                 a_jk = a_jk + ...
%                     g_j/g_k + log(g_k/g_j);
%
%             end
%
%
%             A = A + a_jk;
%
%
%             % ------------------------------------------------
%             % B_jk
%             % ------------------------------------------------
%
%             B_squared = 0;
%
%             for m = 1:R
%
%                 h_m = H_tilde(m,:);
%
%                 g_j = G_cov(j,m);
%                 g_k = G_cov(k,m);
%
%                 s = h_m * V * mu_jk;
%
%                 B_squared = B_squared + ...
%                     abs(s)^2 * g_j / (g_k^2);
%
%             end
%
%             B = sqrt(2 * B_squared + 1e-12);
%
%
%             % ------------------------------------------------
%             % Pairwise exponential Q approximation
%             % ------------------------------------------------
%
%             ratio = A / B;
%
%             L_jk = exp(-tau * ratio^2);
%
%             total_error = total_error + p(j) * L_jk;
%
%
%             % =================================================
%             % Gradient A'_jk
%             % =================================================
%
%             A_grad = zeros(size(V));
%
%             mu_outer = mu_jk * mu_jk';
%
%
%             for m = 1:R
%
%                 h_m = H_tilde(m,:);
%
%                 g_k = G_cov(k,m);
%
%                 A_grad = A_grad + ...
%                     2 * ...
%                     (h_m' * h_m) * ...
%                     V * ...
%                     mu_outer / g_k;
%
%             end
%
%
%             % =================================================
%             % Gradient B'_jk
%             % =================================================
%
%             B_grad = zeros(size(V));
%
%             for m = 1:R
%
%                 h_m = H_tilde(m,:);
%
%                 g_j = G_cov(j,m);
%                 g_k = G_cov(k,m);
%
%                 B_grad = B_grad + ...
%                     (g_j / g_k^2) * ...
%                     (h_m' * h_m) * ...
%                     V * ...
%                     mu_outer;
%
%             end
%
%             B_grad = ...
%                 (2 / B) * B_grad;
%
%
%             % =================================================
%             % Gradient of L_jk
%             %
%             % L'_jk =
%             % -2 tau L A/B^3
%             % (B A' - A B')
%             % =================================================
%
%             L_grad = ...
%                 -2 * tau * L_jk * ...
%                 A / B^3 * ...
%                 (B * A_grad - A * B_grad);
%
%
%             % ------------------------------------------------
%             % Weighted gradient
%             % ------------------------------------------------
%
%             gradV = gradV + p(j) * L_grad;
%
%         end
%
%     end
%
%
%     % --------------------------------------------------------
%     % Gradient descent update
%     % --------------------------------------------------------
%
%     V_new = V - eta * gradV;
%
%
%     % ========================================================
%     % Power projection
%     % ========================================================
%
%     Power_new = 0;
%
%     for j = 1:C
%
%         mu_j = CLS_mean(:,j);
%         Sigma_j = CLS_cov(:,:,j);
%
%         Power_new = Power_new + p(j) * ( ...
%             real(trace( ...
%             V_new * Sigma_j * V_new' ...
%             )) ...
%             + ...
%             real(norm( ...
%             V_new * mu_j ...
%             )^2) ...
%             );
%
%     end
%
%
%     % --------------------------------------------------------
%     % Normalize to satisfy E{||Vx||^2} <= PT
%     % --------------------------------------------------------
%
%
%     alpha_P = sqrt(            P*T / Power_new        );
%
%     V_new = alpha_P * V_new;
%
%
%
%     % Update
%     V = V_new;
%
%     Error(it) = total_error;
%
%
%     % fprintf( ...
%     %     'Iteration %d | MAP surrogate = %.6e | Power = %.4f\n', ...
%     %     it, ...
%     %     total_error, ...
%     %     Power_new ...
%     %     );
%     % errorvec(it) = total_error;
% end
% % close all;plot(errorvec)
%
% end


function V = Our_Encoder( ...
    H, P, CLS_mean, CLS_cov, p, delta_0, ...
    K, N_t_k, N_r, T, eta, Ite)

% ============================================================
% Vectorized MAP precoder optimization
% ============================================================

tau = 0.7;

% ============================================================
% Dimensions
% ============================================================

D = size(CLS_mean, 1);
C = size(CLS_mean, 2);

R = N_r * T;
N = K * N_t_k;

H_tilde = H;

N_T = T * N;

% ============================================================
% Initialization
% ============================================================

V = randn(N_T,D) + 1j*randn(N_T,D);

V = blkdiagonalize(V,K,N_t_k,T,D);

% ============================================================
% Initial power projection
% ============================================================

Power = 0;
Sigma_0 = mean(CLS_cov,3);
for j = 1:C

    mu_j = CLS_mean(:,j);
    % Sigma_j = CLS_cov(:,:,j);
    Vmu = V * mu_j;
    Power = Power + p(j) * ( ...
        real(trace(V * Sigma_0 * V')) + ...
        real(Vmu' * Vmu) );

end
V = sqrt(P*T / Power) * V;
% ============================================================
% Optimization
% ============================================================
Error = zeros(Ite,1);

MU_DIFF = zeros(D,C,C);
for j = 1:C
    for k = 1:C

        if j ~= k
            MU_DIFF(:,j,k) = CLS_mean(:,j) - CLS_mean(:,k);
            aa__jkasdf(j,k) = log(p(j)/p(k)) ;


        end

    end
end


% ============================================================
% Iterations
% ============================================================

for it = 1:Ite

    % ========================================================
    % Step 1: Compute covariance terms
    % ========================================================
    HV = H_tilde * V;



    % H V Sigma
    HV_Sigma = HV *  Sigma_0;

    % Diagonal of H V Sigma V^H H^H
    G_cov1 = real(sum(HV_Sigma .* conj(HV),2)).' + delta_0;
    HVnorm = diag(sqrt(1 ./ G_cov1))*HV;
    HVnorm2 = 2*H_tilde'*diag(1 ./ G_cov1)*HV;

    % ========================================================
    % Step 2: Gradient
    % ========================================================
    gradV = zeros(size(V));
    total_error = 0;
    % fff = 1;
    for j = 1:C

        for k = 1:C

            if j == k
                continue;
            end
            mu_jk = MU_DIFF(:,j,k);
            % =================================================
            % S+log(pj/pk)
            % =================================================
            Sjk = norm(HVnorm*mu_jk)^2;
            Nominator = Sjk + aa__jkasdf(j,k);

            % =================================================
            % S'
            % =================================================
            HV_mu_g = HVnorm2*mu_jk;
            scalar1 = -0.5*tau*exp(-0.5*tau*(Nominator^2/Sjk))*( 1-(aa__jkasdf(j,k)/Sjk)^2  );

            Sjk_prime = HV_mu_g*mu_jk';
            L_grad = scalar1*Sjk_prime;



            % ------------------------------------------------
            % Weighted gradient
            % ------------------------------------------------

            gradV = gradV + p(j)*L_grad;
        end

    end
    % ========================================================
    % Gradient descent update
    % ========================================================

    V_new1 = V - eta*gradV;
    V_new = blkdiagonalize(V_new1,K,N_t_k,T,D);


    % ========================================================
    % Power projection
    % ========================================================

    Power_new = 0;

    for j = 1:C

        mu_j = CLS_mean(:,j);

        Vmu = V_new * mu_j;

        Power_new = Power_new + p(j) * ( ...
            real(trace( ...
            V_new * Sigma_0 * V_new' ...
            )) + ...
            real(Vmu' * Vmu) ...
            );

    end


    % ========================================================
    % Normalize
    % ========================================================

    alpha_P = sqrt(P*T / Power_new);

    V = alpha_P * V_new;


    % ========================================================
    % Save error
    % ========================================================

end

end




function [beta_vec,acc_mcr2] = OurChannel_Receiver( ...
    H,  CLS_cov, CLS_mean, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label,mssag,DetectorType)
for ii = 1:size(CLS_cov,3)
    Traces(ii) = real(trace(CLS_cov(:,:,ii)));
end
epsilon = 1e-6; % trick
acc_mcr2 = 0;
sss = [];
for j = 1:Dataset
    z_cplx = test_feature(:, j);
    % add noise
    n_cplx = delta_0 * sqrt(1/2) * (randn(N_r*T, 1) + 1j*randn(N_r*T, 1));
    C_n_cplx = delta_0^2 * eye(N_r*T);
    signal = V*z_cplx;
    Pow1(j) = norm(signal, 'fro')^2;
    DesiredPower = P_k*K*T;
    y_cplx = sqrt(DesiredPower/(1e-30+Pow1(j)))* H*V*z_cplx + n_cplx;
    G = sqrt(DesiredPower/(1e-30+Pow1(j)))* H*V;
    % analyze_G(G)
    if   sum(abs(z_cplx))~=0
        gammaG= norm(G, 'fro')^2/(size(G,1) * size(G,2));
        betaa = (delta_0/gammaG+mean(Traces))/T/N_r;
        % gammaG+
        sss= [sss, betaa];
    end
    signal2 = sqrt(DesiredPower/(1e-30+Pow1(j)))* V*z_cplx;


    %% Our ML Receiver
    K0 = 0;
    for g = 1:L
        K0 = K0+H*V*CLS_cov(:,:,g)*V'*H'+delta_0*eye(N_r*T);


    end
    KjDiag1 = K0/L;
    KjDiag = real(diag(diag(KjDiag1)));
    for g = 1:L
        diffs = (y_cplx-H*V*CLS_mean(:,g));
        if DetectorType==1
            Kj = H*V*CLS_cov(:,:,g)*V'*H'+delta_0*eye(N_r*T);
            LogLikelihood(g) = real(diffs'*inv(Kj)*diffs+log(det(Kj)));

        elseif DetectorType==0
            LogLikelihood(g) = real(diffs'*inv(KjDiag)*diffs+log(det(KjDiag)));

        end


    end

    [~, pos] = min(LogLikelihood);
    if (pos-1)==test_label(j)
        acc_mcr2 = acc_mcr2 +1;
    end


end
FileName = ['betaBounds' mssag '.mat'];
if ~isfile(FileName)
    min_gamma = inf;
    max_gamma = -inf;
    save(FileName, 'min_gamma','max_gamma')
end

beta_vec = (sss);
load(FileName)
min_gamma1 = min(beta_vec);
max_gamma1 = max(beta_vec);
min_gamma = min(min_gamma1,min_gamma);
max_gamma = max(max_gamma1,max_gamma);

disp([min_gamma,max_gamma])
save(FileName, 'min_gamma','max_gamma')

% update_gamma_histogram(gamma_G_vec, 'gamma_G_hist.mat');
acc_mcr2 = acc_mcr2/Dataset;
end

function lambda = bisection_search_bar(lambda_lb, lambda_ub, P, epsilon, g, Gamma, matrix_form,I_1)

lb = lambda_lb;
ub = lambda_ub;
lambda = lambda_ub;

max_iter = I_1;   % safety cap
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


function fcn_bar = f_bar(lambda, g, Gamma, matrix_form)
%   Detailed explanation goes here

fcn_bar = 0;
if matrix_form == false
    for i = 1:size(Gamma, 1)
        fcn_i = norm(g(i))^2/( (Gamma(i, i) + lambda)^2 );
        fcn_bar = fcn_bar + fcn_i;
    end
else
    for i = 1:size(Gamma, 1)
        fcn_i = abs(g(i))/( (Gamma(i, i) + lambda)^2 );
        fcn_bar = fcn_bar + fcn_i;
    end
end
end
function [mcr2, mcr2_1, mcr2_2] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p)

Dim = size(H, 1);
beta = 1 + alpha*delta_0^2;
AA =  beta*eye(Dim) + alpha*H*V*C*V'*H';
% fprintf('max H = %e\n', max(abs(H(:))));
% fprintf('max V = %e\n', max(abs(V(:))));
% fprintf('max C = %e\n', max(abs(C(:))));
% fprintf('alpha = %e\n', alpha);
% fprintf('beta  = %e\n', beta);

mcr2_1 = log( det(AA ) );

mcr2_2 = 0;
class = size(p, 2);
for j = 1:class
    C_j = CLS_cov(:, :, j);
    mcr2_2 = mcr2_2 + p(j) * ...
        log( det( beta*eye(Dim) + alpha*H*V*C_j*V'*H' ) );
end
% mcr2 = 0.5 * real(mcr2_1 - mcr2_2);
%
% mcr2_1 = 0.5 * real(mcr2_1);
% mcr2_2 = 0.5 * real(mcr2_2);
mcr2 = real(mcr2_1 - mcr2_2);

mcr2_1 = real(mcr2_1);
mcr2_2 = real(mcr2_2);
end
function update_gamma_histogram(gamma_G_new, filename)

% Number of bins
Nbins = 200;

% ---------------------------------------------------------
% Load existing histogram
% ---------------------------------------------------------
if isfile(filename)

    data = load(filename);

    edges = data.edges;
    counts = data.counts;

else

    % First run
    gamma_min = min(gamma_G_new);
    gamma_max = max(gamma_G_new);

    % Give some margin around the initial range
    if gamma_min == gamma_max
        gamma_min = gamma_min / 2;
        gamma_max = gamma_max * 2;
    end

    edges = linspace(gamma_min, gamma_max, Nbins + 1);
    counts = zeros(1, Nbins);

end

% ---------------------------------------------------------
% Expand histogram if new values are outside current range
% ---------------------------------------------------------
new_min = min(gamma_G_new);
new_max = max(gamma_G_new);

while new_min < edges(1)

    width = edges(2) - edges(1);

    edges = [edges(1)-width, edges];

    counts = [0, counts];

end

while new_max >= edges(end)

    width = edges(end) - edges(end-1);

    edges = [edges, edges(end)+width];

    counts = [counts, 0];

end

% ---------------------------------------------------------
% Add new samples
% ---------------------------------------------------------
new_counts = histcounts(gamma_G_new, edges);

counts = counts + new_counts;
% ---------------------------------------------------------
% Plot histogram
% ---------------------------------------------------------
% bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
%
% bar(bin_centers, counts, 1);
%
% xlabel('\gamma_G');
% ylabel('Count');    % ---------------------------------------------------------
% Save updated histogram
% ---------------------------------------------------------
% save(filename, 'edges', 'counts');

end
function [mean_G, zero_mean_ratio, cross_ratio, mean_corr, max_corr] = analyze_G(G)

% ============================================================
% Zero-mean assumption
% ============================================================

mean_G = mean(G(:));

zero_mean_ratio = abs(mean_G) / mean(abs(G(:)));


% ============================================================
% Uncorrelated assumption
% ============================================================

g = G(:);

% Empirical correlation matrix
R = (g * g') / length(g);

% Normalize to correlation coefficients
d = sqrt(abs(diag(R)));

Corr = R ./ (d * d.');

% Off-diagonal elements
off_diag = Corr(~eye(size(Corr)));

% Mean and maximum absolute cross-correlation
mean_corr = mean(abs(off_diag));
max_corr  = max(abs(off_diag));


% ============================================================
% Your requested cross-term / squared-term ratio
% ============================================================
R1 = numel(G)^2-numel(G);
R2 =numel(G);
cross_sum = 0;

for i = 1:length(g)
    for j = 1:length(g)

        if i ~= j
            cross_sum = cross_sum + abs(g(i) * conj(g(j)));
        end

    end
end

squared_sum = sum(abs(g).^2);

cross_ratio = cross_sum / squared_sum/R1*R2;


% ============================================================
% Display
% ============================================================

fprintf('\n===== Analysis of G =====\n');

fprintf('Mean(G) = %.6e + %.6ei\n', ...
    real(mean_G), imag(mean_G));

fprintf('Zero-mean ratio = %.6e\n', ...
    zero_mean_ratio);

fprintf('Cross-term / squared-term ratio = %.6e\n', ...
    cross_ratio);

fprintf('Mean absolute correlation = %.6e\n', ...
    mean_corr);

fprintf('Maximum absolute correlation = %.6e\n', ...
    max_corr);

end
function max_index = get_max_file_index(prefix)

files = dir([prefix '*.mat']);

indices = [];

for i = 1:length(files)

    token = regexp( ...
        files(i).name, ...
        [regexptranslate('escape', prefix) '(\d+)\.mat'], ...
        'tokens' ...
        );

    if ~isempty(token)
        indices(end+1) = str2double(token{1}{1});
    end

end

if isempty(indices)
    error('No files found with prefix: %s', prefix);
end

max_index = max(indices);

end
function V = blkdiagonalize(V, K, N_t_k, T, D)

D_k = D / K;

V_block = zeros(size(V));

for k = 1:K

    rows = (k-1)*N_t_k*T + 1 : k*N_t_k*T;
    cols = (k-1)*D_k + 1 : k*D_k;

    V_block(rows,cols) = V(rows,cols);

end

V = V_block;

end