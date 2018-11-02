function [ predicted ] = predict( weight_hidden_input, weight_output_hidden, X )

totalElements = size(X, 1);

h1 = sigmoid([ones(totalElements, 1) X] * weight_hidden_input');
h2 = sigmoid([ones(totalElements, 1) h1] * weight_output_hidden');
[dummy, predicted] = max(h2, [], 2);

end