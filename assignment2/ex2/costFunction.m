function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
sig = sigmoid(X*theta);
J = (1/m)*((-y'*log(sig))-(-y+1)'*log(1-sig))
grad = (1/m)*X'*(sig-y)








% =============================================================

end
