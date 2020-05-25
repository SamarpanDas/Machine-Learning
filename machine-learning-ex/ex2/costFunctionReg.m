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


z = X * theta;         %getting the value of hypothesis
k = ones(size(z));     
v = -1.*z;
d = k + exp(v);
h = 1./d;

a = y'*log(h);
b = ((ones(size(y)) - y)') * (log(ones(size(h)) - h));
Ji = (-1/m) .* (a+b);   %the usual J without the regularized part


theta1 = theta(1);

initialtheta = zeros(size(theta));
initialtheta = theta; %keeping a copy of th inital theta for future use

theta(1)=[]; %recall we dont add the 1st element of theta durin regularization


Jj= ((theta)' * (theta)).*(lambda/(2*m));  % extra part of J with regularized part

J = Ji + Jj;       %final J



l = (X')*(h - y);
gradi = (l ./ m);  %this is the grad vector without the regularized part

initialtheta(1)= 0;
gradj = (initialtheta) .* (lambda/m); %here I've made the extra part of the regularised grad vector
                               %including the first element of the
                               %vector,although ik that it shouldnt be done
%gradj(1) = (gradj(1) ./ (lambda/m)); %here ive neutralised that 1st elemnt of the 1st element of the extra part

grad = gradi + gradj; % resultant vector








% =============================================================

end
