function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
n=size(theta,1); 
theta1=[theta(2:end)];
h_theta_X= X*theta;
pred= (h_theta_X-y).^2;
Cost= (sum(pred))/(2*m);
Reg=(lambda*sum(theta1.^2))/(2*m);


% abc=(X-y).^2;
% def=sum(abc)/(2*m);

J=Cost + Reg;

grad0= X(:,1)'*(h_theta_X-y)/m;
grad1=X(:,2:n)'*(h_theta_X-y)/m + lambda*theta1/m;

% =========================================================================
grad=[grad0; grad1];
grad = grad(:);

end
