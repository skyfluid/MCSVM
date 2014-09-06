clear
clc
close all
load('CrossValid_MCSVM_10Class.mat');

avgHR = [];
avgvarcov = [];

for nPC=1:50

    p = stat_total.data(nPC).HR(1,:);
    avgHR = [avgHR p'];
    q = stat_total.data(nPC).PCvar;
    avgvarcov = [avgvarcov q'];

end

figure;
boxplot(avgHR);
title(['Accuracy of Multi-Class SVM (199u 50pc)']);
xlabel('Number of PCs'); ylabel('Accuracy (%)');
saveas(gcf, ['./Accuracy_box_50bs_199u_50pc.jpg'], 'jpg');
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
%}
figure;
boxplot(avgvarcov);
title(['Variance Coverage of Multi-Class SVM (199u 50pc)']);
xlabel('Number of PCs'); ylabel('Variance Coverage');
saveas(gcf, ['./Varcov_box_50bs_199u_50pc.jpg'], 'jpg');    


