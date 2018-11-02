function [ trainSet, testSet ] = preprocessData( )

trainingSet = dlmread('optdigits.tra', ',');
testingSet = dlmread('optdigits.tes', ',');

trainTranspose = trainingSet';
testTranspose = testingSet';
k = 1;

for i = 1 : size(trainingSet, 1)
    j = 1;
    if (trainingSet(size(trainingSet, 1) * (size(trainingSet, 2) - 1) + i) == 0 || trainingSet(size(trainingSet, 1) * (size(trainingSet, 2) - 1) + i) == 7 || trainingSet(size(trainingSet, 1) * (size(trainingSet, 2) - 1) + i) == 5)
        trainSet(k, j : j + size(trainingSet, 2) - 1) = trainTranspose(j : j + size(trainingSet, 2) - 1, i);
        k = k + 1;
    end
end

trainSet = trainSet';

k = 1;
for i = 1 : size(testingSet, 1)
    j = 1;
    if or(testingSet(size(testingSet, 1) * (size(testingSet, 2) - 1) + i) == 0, testingSet(size(testingSet, 1) * (size(testingSet, 2) - 1) + i) == 7)
        testSet(k, j : j + size(testingSet, 2) - 1) = testTranspose(j : j + size(testingSet, 2) - 1, i);
        k = k + 1;
    end
end

testSet = testSet';

binaryProcessing = trainSet(1 : size(trainSet, 1) - 1, :);
targetRow = trainSet(size(trainSet, 1), :);

% for i = 1 : size(binaryProcessing, 1) * size(binaryProcessing, 2)
%     if binaryProcessing(i) > 8
%         binaryProcessing(i) = 255;
%     else
%         binaryProcessing(i) = 0;
%     end
% end

trainSet = [binaryProcessing; targetRow];


binaryProcessing = testSet(1 : size(testSet, 1) - 1, :);
targetRow = testSet(size(testSet, 1), :);

% for i = 1 : size(binaryProcessing, 1) * size(binaryProcessing, 2)
%     if binaryProcessing(i) > 8
%         binaryProcessing(i) = 255;
%     else
%         binaryProcessing(i) = 0;
%     end
% end

testSet = [binaryProcessing; targetRow];


end

