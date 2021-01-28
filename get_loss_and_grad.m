function [loss, grads_W1, grads_b1, grads_W2, grads_b2] =  get_loss_and_grad(X, Y, W1, b1, W2, b2, reg, batch_size)
% 前向传播计算损失，再后向传播计算梯度
[N,D] = size(X);
% 前向传播
temp = sigmoid(X * W1 + b1); 
scores = temp * W2 + b2; 
[~,len] = size(scores);
scores = scores - max(scores,[],2);  
sigma = sum(exp(scores),2); 
trans_scores = scores.';
tags = (uint32([0:N-1] *len)' + Y + 1); 
loss = -sum(trans_scores(tags)) + sum(sum(log(sigma))); 
loss = loss / N;
loss = loss + 0.5 * reg * (sum(sum(W1 .* W1) + sum(sum(W2 .* W2))));
temp_scores = exp(scores) ./ sigma;
temp_scores = temp_scores.';
temp_scores(tags) = temp_scores(tags) - 1;
temp_scores = temp_scores.';
temp_scores = temp_scores / N;
grads_W2 = temp' * temp_scores + reg * W2;
grads_b2 = sum(temp_scores, 1);
grads_temp = temp_scores * W2';
hidden = X * W1 + b1;
grads_hidden = grads_temp .* sigmoid(hidden) .* (1 - sigmoid(hidden));
grads_W1 = X' * grads_hidden + reg * W1;
grads_b1 = sum(grads_hidden, 1);

end