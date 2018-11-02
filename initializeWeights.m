function [ weights ] = initializeWeights( layer_previous, layer_post )

epsilon_init = 0.06;

weights = (rand(layer_post, layer_previous + 1) * 2 * epsilon_init) - epsilon_init;

end
