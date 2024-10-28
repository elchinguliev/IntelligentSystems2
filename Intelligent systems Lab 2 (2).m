% Full MATLAB Script for MLP Approximation using Backpropagation

%% Step 1: Generate Input and Target Output
x = 0.1:1/22:1;  % 20 input vectors in the range [0, 1]
%This line generates 20 evenly spaced values for the input x ranging from 0.1 to 1.
% The increment 1/22 ensures that you get 20 values in total.


y = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2; % Target output
%The target output y is calculated using the given formula.
%It simulates a sinusoidal function with two sine components.

%% Step 2: Initialize Parameters of the MLP
input_size = 1;     % One input feature
hidden_size = 6;    % Hidden layer neurons (between 4 and 8, here we use 6)
output_size = 1;    % One output
%input_size is set to 1, meaning the network has one input.
%hidden_size is set to 6, which means there are 6 neurons in the hidden layer.
%output_size is 1, meaning there is one output neuron.


% Initialize random weights and biases
W1 = rand(hidden_size, input_size) * 0.1;  % Weights between input and hidden layer
b1 = rand(hidden_size, 1) * 0.1;           % Bias for hidden layer
W2 = rand(output_size, hidden_size) * 0.1;  % Weights between hidden and output layer
b2 = rand(output_size, 1) * 0.1;            % Bias for output layer
%W1: Weights between the input layer and hidden layer. It is a 6x1 matrix since the hidden layer has 6 neurons and there is 1 input.
%b1: Bias vector for the hidden layer (6x1).
%W2: Weights between the hidden layer and output layer. It is a 1x6 matrix because the output layer has 1 neuron and the hidden layer has 6 neurons.
%b2: Bias for the output layer (1x1).


% Define the hyperbolic tangent activation function and its derivative
tanh_activation = @(z) tanh(z);
tanh_derivative = @(z) 1 - tanh(z).^2;
%These lines define the hyperbolic tangent activation function and its derivative. 
%These will be used to compute activations in the hidden layer and perform backpropagation.


% Define learning rate and number of epochs
lr = 0.01;        % Learning rate
epochs = 10000;   % Number of iterations for training
%The learning rate (lr) controls how much the weights are adjusted at each step.
%The number of epochs (epochs) defines how many times the training loop runs.

%% Step 3: Training the MLP using Backpropagation
for epoch = 1:epochs
    for i = 1:length(x)
        % Forward pass
        z1 = W1 * x(i) + b1;          % Input to hidden layer
        a1 = tanh_activation(z1);     % Activation in hidden layer
        z2 = W2 * a1 + b2;            % Input to output layer
        a2 = z2;                      % Linear activation for output
        %z1 = W1 * x(i) + b1: Computes the input to the hidden layer (z1) as a weighted sum of the input data and the biases.
        %a1 = tanh_activation(z1): Applies the hyperbolic tangent activation function to get the hidden layer activations (a1).
        %z2 = W2 * a1 + b2: Computes the input to the output neuron (z2), as a weighted sum of the hidden layer activations (a1) and the output layer biases (b2).
        %a2 = z2: The output layer uses a linear activation, so a2 is simply z2.

        % Calculate the error
        error = a2 - y(i);
        %The error is calculated as the difference between the predicted output (a2) and the actual target output (y(i))

        % Backpropagation
        delta2 = error;                          % Output layer error
        delta1 = (W2' * delta2) .* tanh_derivative(z1);  % Hidden layer error
        %delta2 = error: The error at the output layer is simply the difference between predicted and actual values.
        %delta1 = (W2' * delta2) .* tanh_derivative(z1): This computes the error at the hidden layer (delta1) using backpropagation. 
        %The error from the output layer is propagated backward through the network, 
        %and the derivative of the tanh function is used to adjust for the hidden layer's non-linearity.

        % Update weights and biases using gradient descent
        W2 = W2 - lr * delta2 * a1';
        b2 = b2 - lr * delta2;
        W1 = W1 - lr * delta1 * x(i);
        b1 = b1 - lr * delta1;
        %The weights W2 and W1 are adjusted based on the errors and activations.
        %The biases b2 and b1 are also updated based on the errors
    end

    % Optional: Display the error every 1000 epochs
    if mod(epoch, 1000) == 0
        disp(['Epoch ' num2str(epoch) ': Error = ' num2str(mean(error.^2))]);
    end
    %This block prints the error every 1000 epochs to monitor the training progress. 
    %The mean squared error is computed and displayed.
end

%% Step 4: Predicting using the Trained MLP
y_pred = zeros(size(x));  % Initialize predicted output array
for i = 1:length(x)
    z1 = W1 * x(i) + b1;        % Input to hidden layer
    a1 = tanh_activation(z1);   % Activation in hidden layer
    z2 = W2 * a1 + b2;          % Input to output layer
    y_pred(i) = z2;             % Linear output
end
%After training, this loop computes the predicted output (y_pred) for 
%each input x(i) by performing the forward pass using the trained weights and biases

%% Step 5: Plot the Results
figure;
plot(x, y, 'b', 'LineWidth', 1.5);  % Plot actual output
hold on;
plot(x, y_pred, 'r--', 'LineWidth', 1.5);  % Plot predicted output
legend('Actual Output', 'MLP Predicted Output');
xlabel('Input x');
ylabel('Output y');
title('MLP Approximation using Backpropagation');
grid on;

