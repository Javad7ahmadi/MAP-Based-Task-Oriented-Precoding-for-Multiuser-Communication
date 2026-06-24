clc;clear;close all;

for iii =1:2
    if iii==1
        mssg = 'MCR2...........';
        load('mcr2_statistics.mat')
        load('mcr2_test_results.mat')

    elseif iii==2
        mssg = 'Proposed...........';
        load('OurStatistics.mat')
        load('OurTest.mat')

    end


    disp(mssg)
    mu = mean(Z, 1);
    Zcc = Z - mu;
    C_xx = (Zcc' * Zcc) / size(Z,1);
    test_label = test_label;%test_labels(1:1000);


    Z_train = Z;              % Ntrain x D  (complex)
    y_train = labels';      % Ntrain x 1


    Z_test = test_feature_cplx;     % Ntest x D (complex)
    y_test = test_label';         % Ntest x 1

    num_classes = 10;

    %% compute class statistics

    CLS_mean = cell(num_classes,1);
    CLS_cov  = cell(num_classes,1);

    for c = 0:num_classes-1

        idx = (y_train == c);

        Zc = Z_train(idx,:);

        mu = mean(Zc,1);

        Xc = Zc - mu;

        C = (Xc' * conj(Xc)) / size(Zc,1);
        C = C + 1e-4 * eye(size(C));
        CLS_mean{c+1} = mu;
        CLS_cov{c+1}  = C;

    end

    %% classification

    pred = zeros(size(y_test));

    for n = 1:size(Z_test,1)

        z = Z_test(n,:);

        score = zeros(num_classes,1);

        for c = 0:num_classes-1

            mu = CLS_mean{c+1};
            C  = eye(size(CLS_cov{c+1},1));%CLS_cov{c+1};
            % C  = CLS_cov{c+1};

            d = z - mu;

            % Mahalanobis-like distance
            score(c+1) = abs(real(d * (pinv(C) * d')));    end

        [~, idx_best] = min(score);

        pred(n) = idx_best - 1;

    end

    %% accuracy

    acc = mean(pred == y_test);
    disp(['Test Accuracy = ', num2str(acc)]);

    %% training accuracy

    pred_train = zeros(size(y_train));

    for n = 1:size(Z_train,1)

        z = Z_train(n,:);

        score = zeros(num_classes,1);

        for c = 0:num_classes-1

            mu = CLS_mean{c+1};
            C  = eye(size(CLS_cov{c+1},1)); % or CLS_cov{c+1}

            d = z - mu;

            score(c+1) = abs(real(d * (pinv(C) * d')));

        end

        [~, idx_best] = min(score);

        pred_train(n) = idx_best - 1;

    end

    train_acc = mean(pred_train == y_train);

    disp(['Training Accuracy = ', num2str(train_acc)]);
    y = labels;     % 1 x N

    classes = unique(y);
    K = length(classes);

    D = size(Z,2);

    %% normalize (IMPORTANT for correlation)
    Z = Z ./ vecnorm(Z,2,2);

    %% -----------------------------
    % PYTHON-CONSISTENT VERSION
    %% -----------------------------

    means = zeros(K,D);

    % class means
    for k = 1:K
        idx = (y == classes(k));
        Zk = Z(idx,:);

        means(k,:) = mean(Zk,1);
    end

    %% intra-class

    intra_vals = zeros(K,1);

    for k = 1:K

        idx = (y == classes(k));
        Zk = Z(idx,:);

        mu = means(k,:);

        corr_vals = abs(Zk * mu') ./ ...
            (vecnorm(Zk,2,2) * norm(mu) + 1e-8);

        intra_vals(k) = mean(corr_vals);
    end

    mean_intra = mean(intra_vals);

    %% inter-class

    inter_vals = [];

    for i = 1:K
        for j = 1:K

            if i ~= j

                mu_i = means(i,:);
                mu_j = means(j,:);

                corr_val = abs(mu_i * mu_j') / ...
                    (norm(mu_i) * norm(mu_j) + 1e-8);

                inter_vals(end+1) = corr_val;

            end
        end
    end

    mean_inter = mean(inter_vals);
    %% summary
    disp(['intra: ', num2str(mean_intra), ', inter: ', num2str(mean_inter), ', ratio: ', num2str(mean_intra/mean_inter)]);
    disp('..............');
    disp('..............');
    disp('..............');
    disp('..............');
end
