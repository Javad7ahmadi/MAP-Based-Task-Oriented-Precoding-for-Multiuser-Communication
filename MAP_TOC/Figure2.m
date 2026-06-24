clc;clear;close all;
addpath(genpath('MatlabMaterials'));
if 0
    load("MCR2_ModelNet10_statistics_complex_24.mat", "feature_cplx", "label",...
        "C_xx_real", "mean_real", "C_xx", "CLS_mean", "CLS_cov", "CLS_rlt");
    load("MCR2_ModelNet10_test_feature_label_complex_24.mat", "p", "test_feature_cplx", "test_label");
else
    iii = 1;
    if iii==1
        mssg = 'MCR2...........';
        load('mcr2_statistics.mat')
        load('mcr2_test_results.mat')

    elseif iii==2
        mssg = 'Proposed...........';
        load('OurStatistics.mat')
        load('OurTest.mat')

    end

    % close all;subplot(2,1,1);plot((ratio_curve));subplot(2,1,2);plot(loss_curve)

    mu = mean(Z, 1);
    Zcc = Z - mu;
    C_xx = (Zcc' * Zcc) / size(Z,1);
end


test_label = test_label;%test_labels(1:1000);


test_feature = test_feature_cplx.';%test_Z(1:1000,:).';


I_1 = 100;
% B = 10*1e3;  % 10 kHz
K = 2;  % num. of devices
N_t_k = 6;  % num. of transmit antennas
N_t = N_t_k * K;
T = 1;  % num. of time slots

Dataset = size(test_feature, 2);
% NMSE_factor = norm(test_feature, "fro")^2/Dataset;
C = C_xx;  % covariance matrix
L = size(CLS_mean,2);  % num. of classes
D = 2*size(test_feature, 1); % the muliply 2 is for real and imag

% delta_0_square = 1e-20*B;  % noise power (W)  -170 dBm/Hz density
% delta_0 = sqrt(delta_0_square);  % noise

% d = 240;  % m
% PL = 32.6 + 36.7*log10(d);  % dB
% PL = 10^(-PL/10);
% delta_0 = delta_0/sqrt(PL);  % normalize path loss into the noise power
delta_0 = 1;  % normalize path loss into the noise power
Var_Vec = [1 2 3  4   5  6   7     9    11   ];
eps = 1e-3;
ITERATIONS = 1000;
Acc_LMMSE  = zeros(ITERATIONS,length(Var_Vec));
Acc_Ours  = zeros(ITERATIONS,length(Var_Vec));
Acc_MCR2  =zeros(ITERATIONS,length(Var_Vec));
Acc_Random  =zeros(ITERATIONS,length(Var_Vec));
    ProposedCOmpelxity=zeros(1,length(Var_Vec)) ;
    MCR2COmpelxity=zeros(1,length(Var_Vec)) ;
    LMMSECOmpelxity=zeros(1,length(Var_Vec));

for iter = 1:ITERATIONS
    for varInx = 1:length(Var_Vec)
     N_r = Var_Vec(varInx);  % num. of receive antennas

        P_k_dBm = 30;  % dBm
        P_k = 1e-3*10.^(P_k_dBm./10);  % W
        P = P_k * K;

        % channel generation
        % kappa = 1;  % Rician factor
        H = [];
        % Ht = [];
        H_k_all = zeros(T*N_r, T*N_t_k, K);
        % H_k_t_all = zeros(N_r, N_t_k, K);
        for k = 1:K
            H_k_t = (randn(N_r, N_t_k) + 1j*randn(N_r, N_t_k)) / sqrt(2);
            % H_k_t_all(:, :, k) = H_k_t;
            %             svd(H_k_t)
            H_k = kron(eye(T), H_k_t);
            H_k_all(:, :, k) = H_k;
            H = [H, H_k];
            % Ht = [Ht, H_k_t];
        end
        disp(['Iter ' num2str(iter) ', var  = ' num2str(varInx)])

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%% Our Precoder %%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        V = Our_Encoder(H,P, CLS_mean, p, delta_0, K, N_t_k,N_r, T);
        %%%%%%MAP receiver %%%%%%%
        Acc_Ours(iter,varInx) = OurChannel_Receiver( ...
            H,  CLS_cov, CLS_mean, CLS_rlt, p, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label);
        Acc_Ours_mean(1,varInx) = mean(Acc_Ours(1:iter,varInx),1);
        Mess{1} = ['Proposed'];
        disp(['Acc Ours = ' num2str(Acc_Ours_mean(1,varInx))])
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%% MCR2 Precoder %%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [V, mcr2_all] = Training_MCR2_precoder( ...
            H, C, CLS_cov, p, delta_0, P_k, K, N_t_k,N_r, T, D,P,L,H_k_all,I_1);
        %%%%%%MAP receiver %%%%%%%
        Acc_MCR2(iter,varInx) = OurChannel_Receiver( ...
            H,  CLS_cov, CLS_mean, CLS_rlt, p, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label);
        Acc_MCR2_mean(1,varInx) = mean(Acc_MCR2(1:iter,varInx),1);
        Mess{2} = ['MCR$^2$'];

        disp(['Acc MCR$^2$  = ' num2str(Acc_MCR2_mean(1,varInx))])
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% LMMSE Precoder%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [V, mse_all] = LMMSE_precoder(H, C, delta_0, P_k, K, N_t_k,N_r, T, D,P,H_k_all,I_1);
        %%%%%%MAP receiver %%%%
        Acc_LMMSE(iter,varInx) = OurChannel_Receiver( ...
            H,  CLS_cov, CLS_mean, CLS_rlt, p, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label);
        % disp(['Iter ' num2str(iter) ', Accuracy LMMSE = ' num2str(mean(Acc_LMMSE(1:iter)))])
        Acc_LMMSE_mean(1,varInx) = mean(Acc_LMMSE(1:iter,varInx),1);
        Mess{3} = ['LMMSE'];
        disp(['Acc LMMSE = ' num2str(Acc_LMMSE_mean(1,varInx))])
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % %%%%%% Random Precoder%%%%%%%%%%%%%%%%%%%%%
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 
        % N_T = K * N_t_k*T;
        % V = randn(N_T, D/2) + 1j * randn(N_T, D/2);
        % %%%%%%MAP receiver %%%%
        % Acc_Random(iter,varInx) = OurChannel_Receiver( ...
        %     H,  CLS_cov, CLS_mean, CLS_rlt, p, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label);
        % % disp(['Iter ' num2str(iter) ', Accuracy LMMSE = ' num2str(mean(Acc_LMMSE(1:iter)))])
        % Acc_Random_mean(1,varInx) = mean(Acc_Random(1:iter,varInx),1);
        % Mess{4} = ['Random'];
        % disp(['Acc Random = ' num2str(Acc_Random_mean(1,varInx))])
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% PLOT %%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp(['......**************************************************************************........'])
        %% Complexity Analysis
    ProposedCOmpelxity(1,varInx) =     L^2*N_t_k* K*T *( D/2 + N_r*T);
    MCR2COmpelxity(1,varInx) =     (D/2)^3+L*N_r^3*T^3+I_1*K*T^3*N_t_k^3*(D/K/2)^3 ;
    LMMSECOmpelxity(1,varInx) =     N_r^3*T^3+I_1*K*T^3*N_t_k^3 ;


    end
close all

figure

% Left y-axis (Accuracy)
yyaxis left

plot(Var_Vec, 1-Acc_Ours_mean, 'LineWidth', 2); hold on
plot(Var_Vec, 1-Acc_MCR2_mean, 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 6);
plot(Var_Vec, 1-Acc_LMMSE_mean, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6);
% plot(Var_Vec, 1-Acc_Random_mean, 'LineWidth', 2);

ylabel('MAP error probability', 'Interpreter', 'latex')

% Right y-axis (Complexity)
yyaxis right
set(gca, 'YScale', 'log')

plot(Var_Vec, ProposedCOmpelxity, 'LineWidth', 2); hold on
plot(Var_Vec, MCR2COmpelxity, 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 6);
plot(Var_Vec, LMMSECOmpelxity, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6);

ylabel('Complexity', 'Interpreter', 'latex')

% Common x-axis
xlabel('Receive antennas ($M$)', 'Interpreter', 'latex')

legend(Mess, 'Interpreter', 'latex', 'Location', 'best')

grid on
box on
xlim([1-.1,11])
drawnow

end















function [V, mcr2_all] = Training_MCR2_precoder( ...
    H, C, CLS_cov, p, delta_0, P_k, K, N_t_k,N_r, T, D,P,L,H_k_all,I_1)

alpha = T*N_r / eps^2;
C_sr = sqrtm(C);

% initialization of V
V_init_k = rand(T*N_t_k, D/(2*K)) + 1j*rand(T*N_t_k, D/(2*K));
V_init = kron(eye(K), V_init_k);
V_init = sqrt(P) * V_init ./ sqrt(trace(V_init*C*V_init'));
V = V_init;

Ite = 30;
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
    Pow1(j) = norm(signal)^2;
    DesiredPower = P_k*K*T;
    y_cplx =sqrt(DesiredPower/Pow1(j))* H*V*z_cplx + n_cplx;
    signal2 = sqrt(DesiredPower/Pow1(j))* V*z_cplx;
    Pow2(j) = norm(signal2)^2;

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

alpha = T*N_r / eps^2;

% initialization of V
V_init_k = rand(T*N_t_k, D/(2*K)) + 1j*rand(T*N_t_k, D/(2*K));
V_init = kron(eye(K), V_init_k);
V_init = sqrt(P) * V_init ./ sqrt(trace(V_init*C*V_init'));
V = V_init;

Ite = 100;
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



function V = Our_Encoder(H, P, CLS_mean, p, delta_0, K, N_t_k, N_r, T)
eta = 0.7; % parameter for approximating qfunction with exponential
Ite = 30;
lr =.1;
% =========================
% Dimensions
% =========================
D = size(CLS_mean, 1);
C = size(CLS_mean, 2);

M_T = N_r * T;
N_T = K * N_t_k*T;

% =========================
% Initialization of V
% =========================
V = randn(N_T, D) + 1j * randn(N_T, D);

% Normalize initial power
x_cov = eye(D);
Powers = sum(abs(V*CLS_mean).^2,1);
V = sqrt(P*T) * V / sqrt(max(Powers));

% =========================
% Noise covariance (isotropic approx)
% =========================
delta_H = delta_0+P/(N_r^2*T)*trace(H*H');
% =========================
% Training / optimization loop (simple GD placeholder)
% =========================

for it = 1:Ite

    gradV = zeros(size(V));

    % =========================
    % Pairwise loss
    % =========================
    for j = 1:C
        for k = 1:C
            if j == k
                continue;
            end

            mu_jk = CLS_mean(:, j) - CLS_mean(:, k);

            s = H * V * mu_jk;
            norm_s2 = real(s' * s);

            % Q surrogate gradient weight
            w = exp(-eta*norm_s2 / (2 * delta_H));
            w_forPlot(k,j) = qfunc(sqrt(norm_s2 / (2 * delta_H)));
            % gradient (chain rule form)
            Delta_grad= p(j)*w  * (H' * s) * (mu_jk');
            % Delta_gradss(:,k,j) = reshape(Delta_grad,1,[]);
            gradV = gradV - Delta_grad;
        end
    end
    Error(it) =sum(sum(w_forPlot));
    % =========================
    % Gradient step
    % =========================
    V = V - lr * gradV;

    % =========================
    % Power projection (global scaling)
    % =========================
    Powers = sum(abs(V*CLS_mean).^2,1);
    V = sqrt(P*T) * V / sqrt(max(Powers));

end






% close all;plot(Error);title(['final error = ' num2str(Error(end)) ', minimum error = ' num2str(min(Error))])
% iiiii = 1;
% for j = 1:C
%     for k = 1:C
%         if j == k
%             continue;
%         end
%         iii(iiiii) = sum(Delta_gradss(:,k,j)-Delta_gradss(:,j,k));
%         iiiii = iiiii+1;
%     end
% end
% sum(iii)

end








function acc_mcr2 = OurChannel_Receiver( ...
    H,  CLS_cov, CLS_mean, CLS_rlt, p, delta_0, P_k, K, N_r, T,L, V,test_feature,Dataset,test_label)
epsilon = 1e-6; % trick
acc_mcr2 = 0;
for j = 1:Dataset
    z_cplx = test_feature(:, j);
    % add noise
    n_cplx = delta_0 * sqrt(1/2) * (randn(N_r*T, 1) + 1j*randn(N_r*T, 1));
    C_n_cplx = delta_0^2 * eye(N_r*T);
    signal = V*z_cplx;
    Pow1(j) = norm(signal)^2;
    DesiredPower = P_k*K*T;
    y_cplx =sqrt(DesiredPower/Pow1(j))* H*V*z_cplx + n_cplx;
    signal2 = sqrt(DesiredPower/Pow1(j))* V*z_cplx;
    Pow2(j) = norm(signal2)^2;



    %% Our ML Receiver
    for g = 1:L
        diffs = (y_cplx-H*V*CLS_mean(:,g));
        Kj = H*V*CLS_cov(:,:,g)*V'*H'+delta_0*eye(N_r*T);
        LogLikelihood(g) = real(diffs'*inv(Kj)*diffs+log(det(Kj)));

    end

    [~, pos] = min(LogLikelihood);
    if (pos-1)==test_label(j)
        acc_mcr2 = acc_mcr2 +1;
    end


end
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


