function [ g ] = sigmoidGradient( z )

t = sigmoid(z);

g = t .* (1 - t);

end