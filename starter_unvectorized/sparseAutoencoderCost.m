function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%----------------
%forward pass and backward pass
p_hat=zeros(hiddenSize,1);
m=zeros(hiddenSize,size(data,2));
for i=1:size(data,2)
    %forward pass  
      x=data(:,i);
      m(:,i)=sigmoid(W1*x+b1);
      p_hat=p_hat+m(:,i);
end

%size(p_hat)
p_hat=(1/size(data,2)).* p_hat; % (hiddenSize x 1)

del_W2=zeros(visibleSize,hiddenSize);
del_W1=zeros(hiddenSize,visibleSize);
del_b2=zeros(visibleSize,1);
del_b1=zeros(hiddenSize,1);


for i=1:size(data,2)
    %forward pass  
      x=data(:,i);
      h=sigmoid((W2*m(:,i))+b2);
      cost=cost+(h-x)'*(h-x);
      
    %backward pass  
      delta_3= (-1.*(x-h)).* (h.*(1-h));
      delta_2= ( (W2'*delta_3) + beta .* (  ((-1*sparsityParam)./p_hat) + ( (1-sparsityParam)./(1-p_hat) ) ) ) .* ( m(:,i) .* (1-m(:,i)));
      del_W2=del_W2 + (delta_3 * m(:,i)');
      del_b2=del_b2 + delta_3;
      del_W1=del_W1 + (delta_2 * x');
      del_b1=del_b1 + delta_2;
      
end

W1grad = (1/size(data,2)).* del_W1 + lambda .* W1;
W2grad = (1/size(data,2)).* del_W2 + lambda .* W2;
b1grad = (1/size(data,2)).* del_b1;
b2grad = (1/size(data,2)).* del_b2;

cost=(1/(2*size(data,2)))*cost + (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))) + beta * ( sparsityParam * sum(log(sparsityParam./p_hat)) + (1-sparsityParam)* sum( log( (1-sparsityParam)./(1-p_hat) ) )   );        


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

