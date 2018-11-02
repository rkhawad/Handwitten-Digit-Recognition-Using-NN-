clc;
close all;
clear all;

[trainingSet, testingSet] = preprocessData();

X_train = trainingSet(1:64, :)';
y_train = trainingSet(65, :)';

for i = 1 : size(y_train, 1)
    if y_train(i) == 0
        y_train(i) = 1;
    else
        y_train(i) = 2;
    end
end


input_layer_size  = 64; 
hidden_layer_size = 12;  
total_labels = 2;

totalInstance = size(X_train, 1);

weight_hidden_input = initializeWeights(input_layer_size, hidden_layer_size);
weight_output_hidden = initializeWeights(hidden_layer_size, total_labels);

[weight_hidden_input, weight_output_hidden] = backPropogate(weight_hidden_input, weight_output_hidden, X_train, y_train);

predicted = predict(weight_hidden_input, weight_output_hidden, X_train);

disp('Training Set Accuracy -- ');

disp(mean(double(predicted == y_train)) * 100);


X_test = testingSet(1:64, :)';
y_test = testingSet(65, :)';

for i = 1 : size(y_test, 1)
    if y_test(i) == 0
        y_test(i) = 1;
    else
        y_test(i) = 2;
    end
end

predicted = predict(weight_hidden_input, weight_output_hidden, X_test);

disp('Testing Set Accuracy -- ');

disp(mean(double(predicted == y_test)) * 100);
