clc
clear
load('./CrossValid_MCSVM_10Class_test_91u.mat');   % require more than 2 mins!
unit_sel = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 30, 33, 39, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 53, 57, 60, 65, 66, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 80, 82, 83, 84, 86, 87, 88, 89, 93, 94, 95, 97, 102, 103, 104, 105, 111, 112, 113, 114, 118, 120, 121, 129, 130, 131, 132, 134, 135, 136, 138, 141, 144, 145, 151, 154, 155, 157, 158, 159, 166, 167, 168, 169, 170, 171, 174, 181, 189, 193];


% for generating the result of all trials
wordtoVerify = {'CLAW' 'CRICKET' 'FLAG' 'FORK' 'LION' 'MEDAL' 'OYSTER' 'SERPENT' 'SHELF' 'SHIRT'};
load('../binMat.mat');

% remove the OTHER words
k = 1;
for i=1:800
    if sum(strcmp(binMat{i}.type, wordtoVerify))
        ccc{k} = binMat{i};
        k = k + 1;
    end
end
binMat = ccc;

trialSet = [];
wordSet = [];
for iTrial=1:length(binMat)
    A = binMat{iTrial}.mat';
    if ~isempty(A)
        A = A(:, unit_sel);    %%% cell selection
    end
    B = binMat{iTrial}.type;
    trialSet = [trialSet; A(:)'];
    wordSet = [wordSet; B];
end
AnsSet = [];
for iAns=1:length(wordSet)
    rst = strcmp(wordSet{iAns}, wordtoVerify);
    if isempty( find(rst) )
        AnsSet = [AnsSet; 0];
    else
        AnsSet = [AnsSet; find(rst)];
    end
end

for nPC=1:91
    stat_total.data(nPC).HRall = []
    stat_total.data(nPC).HRi = []
    
    for iTestSet=1:10
        testBegin = (10-iTestSet)*40+1;
        testEnd = (11-iTestSet)*40;
        testSet = trialSet(testBegin:testEnd,:);
        testSetw = wordSet(testBegin:testEnd,:);
        testSetAns = AnsSet(testBegin:testEnd,:);
        trainingSet = [trialSet(1:testBegin-1,:); trialSet(testEnd+1:end,:)];
        trainingSetw = [wordSet(1:testBegin-1,:); wordSet(testEnd+1:end,:)];
        trainingSetAns = [AnsSet(1:testBegin-1,:); AnsSet(testEnd+1:end,:)];

        coeff = stat_total.data(nPC).pc;  % PCA coeff
        coeff = cell2mat(coeff);

        %%% libSVM
        svmStruct = stat_total.data(nPC).svm{1, iTestSet};

        tSet = trainingSet*coeff(:, 1:nPC);
        [resultSet, accuracy, decision_values] = svmpredict(trainingSetAns, tSet, svmStruct); %, 'showplot',true);
        stat_total.data(nPC).HRi = [stat_total.data(nPC).HRi accuracy];
        tempmat = zeros(length(wordtoVerify), length(wordtoVerify));
        for j=1:length(trainingSetAns)
            tempmat(trainingSetAns(j), resultSet(j)) = tempmat(trainingSetAns(j), resultSet(j)) + 1;
        end
        stat_total.data(nPC).PPi{iTestSet} = tempmat;

        tSet = trialSet*coeff(:, 1:nPC);
        [resultSet, accuracy, decision_values] = svmpredict(AnsSet, tSet, svmStruct); %, 'showplot',true);
        stat_total.data(nPC).HRall = [stat_total.data(nPC).HRall accuracy];
        tempmat = zeros(length(wordtoVerify), length(wordtoVerify));
        for j=1:length(AnsSet)
            tempmat(AnsSet(j), resultSet(j)) = tempmat(AnsSet(j), resultSet(j)) + 1;
        end
        stat_total.data(nPC).PPall{iTestSet} = tempmat;
        
    end    
end

save('CrossValid_MCSVM_10Class_test_50binsize.mat', 'stat_total', '-v7.3')
%}

% plot the Conflict plots
mkdir('./PPforBestSVM_91u');
mkdir('./PPiforBestSVM_91u');
mkdir('./PPallforBestSVM_91u');
for i=1:91
    
    x = stat_total.data(i).HR(1,:);
    fprintf('BestSVM: %d, Max: %f\n', find(x == max(x), 1), max(x));
    
    BestSVMnum = find(x == max(x), 1);
    BestSVM = stat_total.data(i).svm{1, BestSVMnum};
    BestAccuVal(i) = max(x);                % best accuracy for different #PCs
    x = stat_total.data(i).HRi(1,:);
    BestAccuVali(i) = x(1, BestSVMnum);     
    x = stat_total.data(i).HRall(1,:);
    BestAccuValall(i) = x(1, BestSVMnum);   
    
    imagesc(stat_total.data(i).PP{BestSVMnum});
    colorbar(); set(gca,'XAxisLocation', 'top');
    xlabel('SVM prediction'); ylabel('Correct Answer');
    title(['SVM #' num2str(BestSVMnum) ' for #PC=' num2str(i) '; Categories: CLAW,CRICKET,FLAG,FORK,LION,MEDAL,OYSTER,SERPENT,SHELF,SHIRT']);
    saveas(gcf, ['./PPforBestSVM/PP' num2str(i) '_91u.jpg'], 'jpg');
    
    imagesc(stat_total.data(i).PPi{BestSVMnum});
    colorbar(); set(gca,'XAxisLocation', 'top');
    xlabel('SVM prediction'); ylabel('Correct Answer');
    title(['SVM #' num2str(BestSVMnum) ' for #PC=' num2str(i) '; Categories: CLAW,CRICKET,FLAG,FORK,LION,MEDAL,OYSTER,SERPENT,SHELF,SHIRT']);
    saveas(gcf, ['./PPiforBestSVM/PPi' num2str(i) '_91u.jpg'], 'jpg');

    imagesc(stat_total.data(i).PPall{BestSVMnum});
    colorbar(); set(gca,'XAxisLocation', 'top');
    xlabel('SVM prediction'); ylabel('Correct Answer');
    title(['SVM #' num2str(BestSVMnum) ' for #PC=' num2str(i) '; Categories: CLAW,CRICKET,FLAG,FORK,LION,MEDAL,OYSTER,SERPENT,SHELF,SHIRT']);
    saveas(gcf, ['./PPallforBestSVM/PPall' num2str(i) '_91u.jpg'], 'jpg');
    
    % generate the weight of PCs
    %x = BestSVM.
end

plot(BestAccuVal);
xlabel('#PCs'); ylabel('Accuracy (%)');
title('Accuracy (test) of the Best SVM');
saveas(gcf, ['./Accu_Best_91u.jpg'], 'jpg');

plot(BestAccuVali);
xlabel('#PCs'); ylabel('Accuracy (%)');
title('Accuracy (training) of the Best SVM');
saveas(gcf, ['./Accu_Best_i_91u.jpg'], 'jpg');

plot(BestAccuValall);
xlabel('#PCs'); ylabel('Accuracy (%)');
title('Accuracy (overall) of the Best SVM');
saveas(gcf, ['./Accu_Best_all_91u.jpg'], 'jpg');
