clc
clear
close all

load('../binMat.mat')

wordtoVerify = {'CLAW' 'CRICKET' 'FLAG' 'FORK' 'LION' 'MEDAL' 'OYSTER' 'SERPENT' 'SHELF' 'SHIRT'};
unit_sel = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 30, 33, 39, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 53, 57, 60, 65, 66, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 80, 82, 83, 84, 86, 87, 88, 89, 93, 94, 95, 97, 102, 103, 104, 105, 111, 112, 113, 114, 118, 120, 121, 129, 130, 131, 132, 134, 135, 136, 138, 141, 144, 145, 151, 154, 155, 157, 158, 159, 166, 167, 168, 169, 170, 171, 174, 181, 189, 193];

VarCovThreshold = 0.95;
mkdir('PCA_SVM/')

for i=1:800
    if sum(strcmp(binMat{i}.type, wordtoVerify));
        ccc(i) = binMat{i};
    end
end
binMat = ccc;

trialSet = [];
wordSet = [];

for iTrial=1:length(binMat)
    A = binMat(iTrial).mat';
    if ~isempty(A)
        A = A(:, unit_sel);    %%% cell selection
    end
    %A = binMat(iTrial).mat';
    B = binMat(iTrial).type;
    trialSet = [trialSet; A(:)'];
    wordSet = [wordSet; B];
end

outfile = fopen('MCSVM.log','w');

AnsSet = [];

for iAns=1:length(wordSet)
    rst = strcmp(wordSet{iAns}, wordtoVerify);
    if isempty( find(rst) )
        AnsSet = [AnsSet; 0];
    else
        AnsSet = [AnsSet; find(rst)];
    end
end
        
stat_total = {};

overallPP = zeros(length(wordtoVerify), length(wordtoVerify));
for nPC=1:300
    stat_total.data(nPC).HR = []
    
    SVM_candidate = [];
    
    for iTestSet=1:10
        testBegin = (10-iTestSet)*40+1;
        testEnd = (11-iTestSet)*40;
        testSet = trialSet(testBegin:testEnd,:);
        testSetw = wordSet(testBegin:testEnd,:);
        testSetAns = AnsSet(testBegin:testEnd,:);
        trainingSet = [trialSet(1:testBegin-1,:); trialSet(testEnd+1:end,:)];
        trainingSetw = [wordSet(1:testBegin-1,:); wordSet(testEnd+1:end,:)];
        trainingSetAns = [AnsSet(1:testBegin-1,:); AnsSet(testEnd+1:end,:)];

        local_best_score = 0;
        
        for iWorkSet=1:9
            verifyBegin = (9-iWorkSet)*40+1;
            verifyEnd = (10-iWorkSet)*40;
            verifySet = trainingSet(verifyBegin:verifyEnd,:);
            verifySetw = trainingSetw(verifyBegin:verifyEnd,:);
            verifySetAns = trainingSetAns(verifyBegin:verifyEnd,:);
            workSet = [trainingSet(1:verifyBegin-1,:); trainingSet(verifyEnd+1:end,:)];
            workSetw = [trainingSetw(1:verifyBegin-1,:); trainingSetw(verifyEnd+1:end,:)];
            workSetAns = [trainingSetAns(1:verifyBegin-1,:); trainingSetAns(verifyEnd+1:end,:)];

            [coeff, score, latent] = pca(workSet);  % PCA

            %varCov = cumsum(latent)./sum(latent);    % find the variance coverage for PCs
            %nPC = find(varCov > VarCovThreshold, 1);
            scoreSel = (workSet) * coeff(:, 1:nPC);
            
            %%% libSVM
%           options.MaxIter 
            %svmStruct = svmtrain(scoreSel, distingSet); %, 'showplot',true);
            svmStruct = svmtrain(workSetAns, scoreSel); %, 'showplot',true);
            %resultSet = svmclassify(svmStruct, verifySet*coeff(:, 1:nPC)); %, 'showplot',true);
            tSet = verifySet*coeff(:, 1:nPC);
            [resultSet, accuracy, decision_values] = svmpredict(verifySetAns, tSet, svmStruct); %, 'showplot',true);
            %decision_values
            
            if (accuracy(1) > local_best_score)
                local_best_score = accuracy(1);
                SVM_candidate{iTestSet}.svm = svmStruct;
                SVM_candidate{iTestSet}.pc = coeff(:, 1:nPC);
                pcvar = (cumsum(latent) ./ sum(latent));
                SVM_candidate{iTestSet}.pcvar = pcvar(nPC);
            end
        end
        [resultSet, accuracy, decision_values] = svmpredict(testSetAns, testSet*SVM_candidate{iTestSet}.pc, SVM_candidate{iTestSet}.svm);
                        
        fprintf(outfile, 'SVM-%2d, #TestVec: %d, #HR: %f, #PC:%d\n', iTestSet, length(resultSet), accuracy(1), nPC);
                
        stat_total.data(nPC).HR = [stat_total.data(nPC).HR accuracy];
        stat_total.data(nPC).HRCLAW(iTestSet)      = sum(resultSet(find(testSetAns == 1)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRCRICKET(iTestSet)   = sum(resultSet(find(testSetAns == 2)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRFLAG(iTestSet)      = sum(resultSet(find(testSetAns == 3)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRFORK(iTestSet)      = sum(resultSet(find(testSetAns == 4)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRLION(iTestSet)      = sum(resultSet(find(testSetAns == 5)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRMEDAL(iTestSet)     = sum(resultSet(find(testSetAns == 6)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HROYSTER(iTestSet)    = sum(resultSet(find(testSetAns == 7)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRSERPENT(iTestSet)   = sum(resultSet(find(testSetAns == 8)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRSHELF(iTestSet)     = sum(resultSet(find(testSetAns == 9)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).HRSHIRT(iTestSet)     = sum(resultSet(find(testSetAns == 10)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data(nPC).PCvar(iTestSet)        = SVM_candidate{iTestSet}.pcvar;
        stat_total.data(nPC).resultSet{iTestSet}    = resultSet;
        stat_total.data(nPC).ansSet{iTestSet}       = testSetAns;
        stat_total.data(nPC).svm{iTestSet}          = SVM_candidate{iTestSet}.svm;
        stat_total.data(nPC).pc{iTestSet}           = SVM_candidate{iTestSet}.pc;
        
        tempmat = zeros(length(wordtoVerify), length(wordtoVerify));
        for j=1:length(testSetAns)
            tempmat(testSetAns(j), resultSet(j)) = tempmat(testSetAns(j), resultSet(j)) + 1;
            overallPP(testSetAns(j), resultSet(j)) = overallPP(testSetAns(j), resultSet(j)) + 1;
        end
        stat_total.data(nPC).PP{iTestSet} = tempmat;
        
    end
    stat_total.avgHR(nPC) = mean(stat_total.data(nPC).HR(1,:), 2);
    stat_total.avgHRCLAW(nPC)     = mean(stat_total.data(nPC).HRCLAW, 2);
    stat_total.avgHRCRICKET(nPC)  = mean(stat_total.data(nPC).HRCRICKET, 2);
    stat_total.avgHRFLAG(nPC)     = mean(stat_total.data(nPC).HRFLAG, 2);
    stat_total.avgHRFORK(nPC)     = mean(stat_total.data(nPC).HRFORK, 2);
    stat_total.avgHRLION(nPC)     = mean(stat_total.data(nPC).HRLION, 2);
    stat_total.avgHRMEDAL(nPC)    = mean(stat_total.data(nPC).HRMEDAL, 2);
    stat_total.avgHROYSTER(nPC)   = mean(stat_total.data(nPC).HROYSTER, 2);
    stat_total.avgHRSERPENT(nPC)  = mean(stat_total.data(nPC).HRSERPENT, 2);
    stat_total.avgHRSHELF(nPC)    = mean(stat_total.data(nPC).HRSHELF, 2);
    stat_total.avgHRSHIRT(nPC)    = mean(stat_total.data(nPC).HRSHIRT, 2);
    stat_total.avgPCvar(nPC) = mean(stat_total.data(nPC).PCvar, 2);
    
    stat_total.overallPP = overallPP;
    
    save('CrossValid_MCSVM_10Class_test_91u.mat', 'stat_total')
end

fclose(outfile);


save('CrossValid_MCSVM_10Class_test_91u.mat', 'stat_total', '-v7.3')

set(0, 'DefaultFigureVisible', 'on')
