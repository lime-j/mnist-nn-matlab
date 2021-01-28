function result = predict_from_file(dist,W1, b1, W2, b2)
X_val = imresize(rgb2gray(imread(dist)), [28 28]); % 读取图片，转换成灰度图，图片大小变成28 * 28
X_val = flatten(dlarray(double(X_val)./255, "SSCB"), 28 * 28, 1)'; % 归一化，转换为dlarray(matlab深度学习的数据类型)，把矩阵转化成N * 784的格式
result = extractdata(predict(X_val, W1, b1, W2, b2)); % 计算结果，并把结果从dlarray转换成正常的变量
end