function [ weight_hidden_input, weight_output_hidden ] = backPropogate(weight_hidden_input, weight_output_hidden, X, y )

totalElements = size(X, 1);

eta = 0.15;

D1 = zeros(size(weight_hidden_input));
D2 = zeros(size(weight_output_hidden));

y_matrix = zeros(size(y, 1), 2);
for i = 1 : totalElements
    if y(i) == 1
        y_matrix(i, 1) = 1;
    else
        y_matrix(i, 2) = 1;
    end
end

X = [ones(totalElements, 1) X];


for iteration = 1 : 10
    
    RI = randperm(totalElements);
    
    for m = 1 : totalElements
        
        re = RI(m);
        
        a1 = X(re, :);
        
        z2 = a1 * weight_hidden_input';
        
        a2 = sigmoid(z2);
        
        a2 = [1 a2];
        
        z3 = a2 * weight_output_hidden';
        
        h = sigmoid(z3);
        
        a3 = h;
        if a3(1) > a3(2)
            a3(1) = 1;
            a3(2) = 0;
        else
            a3(1) = 0;
            a3(2) = 1;
        end
        
        delta3 = ([y_matrix(re, 1) y_matrix(re, 2)] - a3) .* sigmoidGradient(z3);
        
        delta2 = (delta3 * weight_output_hidden(:, 2:end)) .* sigmoidGradient(z2);
        
        
        D2 = eta * delta3' * a2;
        D1 = eta * delta2' * a1;
        
        weight_hidden_input = weight_hidden_input + D1;
        weight_output_hidden = weight_output_hidden + D2;
        
    end
    
    
end

end
