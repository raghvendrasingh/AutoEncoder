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
% follows the nota;tion convention of the lecture notes. 
p=sparsityParam;
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 



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
    %forward pass  
      a2=sigmoid(W1*data+repmat(b1,1,size(data,2)));
      p_hat=(1/size(data,2)).* sum(a2,2);
      a3=sigmoid(W2*a2+repmat(b2,1,size(data,2)));
      
      cost=(1/(2*size(data,2)))*sum( sum ( (a3-data).*(a3-data) ) ) +... 
           (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))) +... 
           beta * ( p * sum(log(p./p_hat)) + (1-p)* sum( log( (1-p)./(1-p_hat) ) ) );        


    %backward pass  
      delta_3= (-1.*(data-a3)).* (a3.*(1-a3));
      a=(W2'*delta_3);
      b=((-1*p)./p_hat);
      c=( (1-p)./(1-p_hat) );
      d=(beta .* (b+c));
      e=repmat(d,1,size(data,2));
      delta_2= ( a + e).* ( a2 .* (1-a2));
      del_W2=delta_3 * a2';
      del_b2=sum(delta_3,2);
      del_W1=delta_2 * data';
      del_b1=sum(delta_2,2);
      

W1grad = (1/size(data,2)).* del_W1 + lambda .* W1;
W2grad = (1/size(data,2)).* del_W2 + lambda .* W2;
b1grad = (1/size(data,2)).* del_b1;
b2grad = (1/size(data,2)).* del_b2;

%cost=(1/(2*size(data,2)))*cost + (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))) + beta * ( sparsityParam * sum(log(sparsityParam./p_hat)) + (1-sparsityParam)* sum( log( (1-sparsityParam)./(1-p_hat) ) )   );        


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

