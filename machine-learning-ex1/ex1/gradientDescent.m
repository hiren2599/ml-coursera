function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);



for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    n1=0;
    h=theta'*X';
    for i=1:m
      n1=n1+((h(i)-y(i)));
    endfor
    n1=n1/m;
    n2=0;
    j=m+1;
    for i=1:m
      n2=n2+((h(i)-y(i))*X(j));
      j=j+1;
    endfor

    n2=n2/m;
   
    theta(1)=theta(1)-alpha*n1;
    theta(2)=theta(2)-alpha*n2;


    fprintf('Theta found by gradient descent:\n');
    fprintf('%f\n', theta);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
