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

h_theta_x = sigmoid(X*theta);
theta1 = [theta(2);theta(3)];
[J0,grad0]=costFunction(theta,X,y);
J= J0 + (lambda*sum((power(theta1,2)))/(2*m));
% J = -(y'*log(h_theta_x)+(1-y)'*log(1- h_theta_x))/m + (lambda*sum((power(theta1,2)))/(2*m));
% grad(1) = X(:,1)'*(h_theta_x - y)/m;
% 
% for i_grad=2:size(theta)
%     grad(i_grad) = X(:,i_grad)'*(h_theta_x - y)/m + (lambda*theta(i_grad))/m;
% end
% =============================================================
grad= grad0 + [0;(lambda*theta(2:end))/m];
end
