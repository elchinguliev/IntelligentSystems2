% Script 2: Surface Approximation with Two Inputs
% Define input grid and target output
[X1, X2] = meshgrid(0:0.1:1, 0:0.1:1);  % Grid of two input variables
Y = (1 + 0.6 * sin(2 * pi * X1 / 0.7) + 0.3 * sin(2 * pi * X2)) / 2;  % Target output
X1 = X1(:);  % Flatten input grid for training
X2 = X2(:);
Y = Y(:);

% Define MLP structure
input_size = 2;          % Two inputs
hidden_size = 6;         % 6 neurons in the hidden layer
output_size = 1;         % One output

% Initialize random weights and biases
W1 = rand(hidden_size, input_size) * 0.1;  % Input to hidden weights
b1 = rand(hidden_size, 1) * 0.1;           % Hidden layer biases
W2 = rand(output_size, hidden_size) * 0.1; % Hidden to output weights
b2 = rand(output_size, 1) * 0.1;           % Output layer bias

% Activation function and its derivative
tanh_activation = @(z) tanh(z);
tanh_derivative = @(z) 1 - tanh(z).^2;

% Training parameters
lr = 0.01;           % Learning rate
epochs = 10000;      % Training epochs

% Training loop with backpropagation
for epoch = 1:epochs
    for i = 1:length(X1)
        % Forward pass
        x_input = [X1(i); X2(i)];
        z1 = W1 * x_input + b1;         % Input to hidden layer
        a1 = tanh_activation(z1);       % Hidden layer activation
        z2 = W2 * a1 + b2;              % Input to output layer
        a2 = z2;                        % Linear output
        
        % Calculate error
        error = a2 - Y(i);
        
        % Backpropagation
        delta2 = error;                          % Output layer error
        delta1 = (W2' * delta2) .* tanh_derivative(z1);  % Hidden layer error
        
        % Update weights and biases
        W2 = W2 - lr * delta2 * a1';
        b2 = b2 - lr * delta2;
        W1 = W1 - lr * delta1 * x_input';
        b1 = b1 - lr * delta1;
    end
    
    % Display error every 1000 epochs
    if mod(epoch, 1000) == 0
        disp(['Epoch ' num2str(epoch) ': Error = ' num2str(mean(error.^2))]);
    end
end

% Prediction after training
Y_pred = zeros(size(Y));
for i = 1:length(X1)
    x_input = [X1(i); X2(i)];
    z1 = W1 * x_input + b1;
    a1 = tanh_activation(z1);
    z2 = W2 * a1 + b2;
    Y_pred(i) = z2;
end

% Reshape results for plotting
X1_grid = reshape(X1, [11, 11]);
X2_grid = reshape(X2, [11, 11]);
Y_grid = reshape(Y, [11, 11]);
Y_pred_grid = reshape(Y_pred, [11, 11]);

% Plot the surface approximation results
figure;
surf(X1_grid, X2_grid, Y_grid, 'FaceAlpha', 0.5);  % Target output
hold on;
surf(X1_grid, X2_grid, Y_pred_grid, 'FaceAlpha', 0.5);  % Predicted output
legend('Actual Output Surface', 'MLP Predicted Surface');
xlabel('Input X1');
ylabel('Input X2');
zlabel('Output Y');
title('Surface Approximation using MLP');
grid on;
