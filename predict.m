function result = predict(X, W1, b1, W2, b2)
%PREDICT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
temp = sigmoid(X * W1 + b1); %��һ�� -> �ڶ��� -> sigmoid�����
scores = temp * W2 + b2; % �ڶ��� -> ������
[~, result] = max(scores,[],2);
result = result - 1;
end

