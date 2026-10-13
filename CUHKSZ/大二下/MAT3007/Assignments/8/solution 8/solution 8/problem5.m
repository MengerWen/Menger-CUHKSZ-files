% Define the problem
A = [2, 0; 0, 5]; % Matrix A
f = @(x) x' * A * x; % Objective function
grad_f = @(x) 2 * A * x; % Gradient of the function
x0 = [1; 1]; % Initial point

% Constants for Backtracking Line Search
gamma = 0.5;  % Armijo parameter
sigma = 0.2;  % Step size reduction factor

% Run all three step size strategies
num_iterations = 10;

% 1. Gradient Descent with Constant Step Size (alpha = 0.1)
fprintf('Gradient Descent with Constant Step Size:\n');
alpha_const = 0.1;
x_const = x0;
for k = 1:num_iterations
    grad = grad_f(x_const);
    x_const = x_const - alpha_const * grad;
    fprintf('Iteration %d: x = [%f, %f], f(x) = %f\n', k, x_const(1), x_const(2), f(x_const));
end

fprintf('\n');

% 2. Gradient Descent with Exact Line Search
fprintf('Gradient Descent with Exact Line Search:\n');
x_exact = x0;
for k = 1:num_iterations
    grad = grad_f(x_exact);
    % Compute exact step size: alpha = grad' * grad / (2 * grad' * A * grad)
    alpha_exact = (grad' * grad) / (2 * grad' * A * grad);
    x_exact = x_exact - alpha_exact * grad;
    fprintf('Iteration %d: x = [%f, %f], f(x) = %f\n', k, x_exact(1), x_exact(2), f(x_exact));
end

fprintf('\n');

% 3. Gradient Descent with Backtracking Line Search
fprintf('Gradient Descent with Backtracking Line Search:\n');
x_backtrack = x0;
for k = 1:num_iterations
    grad = grad_f(x_backtrack);
    alpha_back = 1; % Start with initial step size
    % Perform backtracking line search
    while f(x_backtrack - alpha_back * grad) > f(x_backtrack) - gamma * alpha_back * norm(grad)^2
        alpha_back = sigma * alpha_back; % Reduce step size
    end
    x_backtrack = x_backtrack - alpha_back * grad;
    fprintf('Iteration %d: x = [%f, %f], f(x) = %f\n', k, x_backtrack(1), x_backtrack(2), f(x_backtrack));
end

fprintf('\n');

% 4. Gradient Descent with Backtracking Line Search without restarting
% step size
fprintf('Gradient Descent with Backtracking Line Search without restarting step size:\n');
x_backtrack = x0;
alpha_back = 1;
for k = 1:num_iterations
    grad = grad_f(x_backtrack);
    % Perform backtracking line search
    while f(x_backtrack - alpha_back * grad) > f(x_backtrack) - gamma * alpha_back * norm(grad)^2
        alpha_back = sigma * alpha_back; % Reduce step size
    end
    x_backtrack = x_backtrack - alpha_back * grad;
    fprintf('Iteration %d: x = [%f, %f], f(x) = %f\n', k, x_backtrack(1), x_backtrack(2), f(x_backtrack));
end