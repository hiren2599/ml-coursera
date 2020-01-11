function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=theta'*X';
g=sigmoid(z);
g=g';


for i=1:m
  if y(i)==1
    J=J-log(g(i));
  else
    J=J-log(1-g(i));
  endif
endfor

J=J/m;

t=0;
for i=2:length(theta)
  t=t+(theta(i)^2);
endfor
t=lambda*t/2;
t=t/m;
J=J+t;

size(grad);

theta=lambda*theta;
grad(1)=((((g-y)')*(X(:,1))));
for i=2:length(grad) 
    grad(i)=((((g-y)')*(X(:,i)))+theta(i));
endfor

for i=1:length(grad) 
  grad(i)=(grad(i)/m);
endfor








% =============================================================

end
