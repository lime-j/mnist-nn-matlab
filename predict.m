function result = predict(X, W1, b1, W2, b2)
%PREDICT 此处显示有关此函数的摘要
%   此处显示详细说明
temp = sigmoid(X * W1 + b1); %第一层 -> 第二层 -> sigmoid激活函数
scores = temp * W2 + b2; % 第二层 -> 第三层
[~, result] = max(scores,[],2);
result = result - 1;
end

