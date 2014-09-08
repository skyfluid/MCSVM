
clear
clc
close all

def_binsize = 200;

figure(1)
set(figure(1), 'Position', [0 0 3840 2160])

% 199 units
load(['./CrossValid_MCSVM_10Class_test_' int2str(def_binsize) 'binsize_199u.mat']);
avgHR = [];
avgvarcov = [];

for nPC=1:50
    p = stat_total.data(nPC).HR(1,:);
    avgHR = [avgHR p'];
    q = stat_total.data(nPC).PCvar;
    avgvarcov = [avgvarcov q'];
end

H(1) = subplot(1, 3, 1);
boxplot(avgHR);
title(['Accuracy of Multi-Class SVM (199u 50pc)']);
xlabel('Number of PCs'); ylabel('Accuracy (%)');
%ylim([0 50])
ylim([5 65])

% 91 units
load(['./CrossValid_MCSVM_10Class_test_' int2str(def_binsize) 'binsize_91u.mat']);
avgHR = [];
avgvarcov = [];

for nPC=1:50
    p = stat_total.data(nPC).HR(1,:);
    avgHR = [avgHR p'];
    q = stat_total.data(nPC).PCvar;
    avgvarcov = [avgvarcov q'];
end

H(2) = subplot(1, 3, 2);
boxplot(avgHR);
title(['Accuracy of Multi-Class SVM (91u 50pc)']);
xlabel('Number of PCs'); ylabel('Accuracy (%)');
%ylim([0 50])
ylim([5 65])

% 16 units
load(['./CrossValid_MCSVM_10Class_test_' int2str(def_binsize) 'binsize_16u.mat']);
avgHR = [];
avgvarcov = [];
for nPC=1:min(50, length(stat_total.data))
    p = stat_total.data(nPC).HR(1,:);
    avgHR = [avgHR p'];
    q = stat_total.data(nPC).PCvar;
    avgvarcov = [avgvarcov q'];
end

% adjust space

H(3) = subplot(1, 3, 3);
boxplot(avgHR);
title(['Accuracy of Multi-Class SVM (16u 50pc)']);
xlabel('Number of PCs'); ylabel('Accuracy (%)');
%ylim([0 50])
ylim([5 65])

%{
P = get(H(1), 'pos');
P(3) = P(3) + 0.01;
P(4) = P(4) + 0.01;
set(H, 'pos', P);

P = get(H(2), 'pos');
P(3) = P(3) + 0.01;
P(4) = P(4) + 0.01;
set(H, 'pos', P);

P = get(H(3), 'pos');
P(3) = P(3) + 0.01;
P(4) = P(4) + 0.01;
set(H, 'pos', P);
%}

% output to file
saveas(gcf, ['./Accuracy_box_' int2str(def_binsize) 'bs_same_scale.jpg'], 'jpg');


%{
figure;
boxplot(avgpHR);
title(['Accuracy (true) of Multi-Class SVM (199u 28pc)']);
xlabel('Number of PCs'); ylabel('Accuracy: hit/ (hit+miss)');
saveas(gcf, ['./PCA_SVM/AccuracyP_box.jpg'], 'jpg');

figure;
boxplot(avgnHR);
title(['Accuracy (false) of Multi-Class SVM (199u 28pc)']);
xlabel('Number of PCs'); ylabel('Accuracy: CR/(CR+FA)');
saveas(gcf, ['./PCA_SVM/AccuracyN_box.jpg'], 'jpg');

figure;
boxplot(avgvarcov);
title(['Variance Coverage of Multi-Class SVM (199u 50pc)']);
xlabel('Number of PCs'); ylabel('Variance Coverage');
saveas(gcf, ['./Varcov_box_50bs_199u_50pc.jpg'], 'jpg');    
%}

