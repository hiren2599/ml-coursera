function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%input_layer_size
%hidden_layer_size
%size(Theta1)
%size(Theta2)
%size(X)
%size(y)
%num_labels
%m
temp_a1=X;
X=[ones(m,1) X];
%size(X)
z2=X*Theta1';     
a2=sigmoid(X*Theta1');
temp_a2=a2;
a2=[ones(size(a2,1),1) a2];
%size(a2)

z3=a2*Theta2';
a3=sigmoid(a2*Theta2');
%size(a3)

%mapping y into matrix
for i=1:m
  temp=y(i);
    for j=1:num_labels
      if(temp==j)
        y(i,j)=1;
      else
        y(i,j)=0;
      endif
    endfor
endfor
%size(y)
for i=1:m
  for j=1:num_labels
     J=J+y(i,j)*log(a3(i,j))+(1-y(i,j))*log(1-a3(i,j));
  endfor
endfor
J=-J;
J=J/m;


%regularization using for loop
temp=0;
for L=1:2
  if L==1
    for i=1:input_layer_size
      for j=1:hidden_layer_size
        if i==1
          temp=temp+0;
        else
          temp=temp+Theta1(j,i)^2;
        endif
      endfor
    endfor
  else
    for i=1:hidden_layer_size+1
      for j=1:num_labels
        if i==1
          temp=temp+0;
        else
          temp=temp+Theta2(j,i)^2;
        endif
      endfor
    endfor
  endif
endfor

%regularization using matrix manipulation
th1=Theta1;
th2=Theta2;
th1(:,1)=0;
th2(:,1)=0;
th1=th1.^2;
th2=th2.^2;
t=sum(sum(th1))+sum(sum(th2));



t=lambda*t;
t=t/2;
t=t/m;

J=J+t;%total cost including regularization


%---------------------------------------------------------------------------
%gradient

%size(Theta1_grad)
%size(Theta2_grad)

a1=X;
sd3=a3-y;

%removing 1st row
t1=Theta1(:,2:end);
t2=Theta2(:,2:end);
sd2=(sd3*t2).*sigmoidGradient(z2);

cd1=sd2'*a1;
cd2=sd3'*a2;

%size(cd1)
%size(cd2)
Theta1_grad=Theta1_grad+cd1/m;
Theta2_grad=Theta2_grad+cd2/m;


t1=[zeros(size(t1,1),1) t1];
t2=[zeros(size(t2,1),1) t2];


Theta1_grad = Theta1_grad + lambda/m * t1;
Theta2_grad = Theta2_grad + lambda/m * t2;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
