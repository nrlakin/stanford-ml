function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

m = length(y); % number of training examples
n = size(X,2);
% You need to return the following variables correctly
J = 0;
theta1=zeros(size(theta));
theta1=theta;
theta1(1)=0;
grad = zeros(size(theta));
sig = sigmoid(X*theta);
J = (1/m)*((-y'*log(sig))-(-y+1)'*log(1-sig));
J +=(lambda/(2*m))*(theta1'*theta1);
grad = (1/m)*X'*(sig-y)+(lambda/m)*theta1;

grad = grad(:);

end
