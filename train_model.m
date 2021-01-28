%https://ww2.mathworks.cn/help/deeplearning/ug/data-sets-for-deep-learning.html
fprintf("Loading MNIST")
XTrain = flatten(processImagesMNIST('./dataset/train-images.idx3-ubyte'), 28 * 28, 60000)';
YTrain = processLabelsMNIST('./dataset/train-labels.idx1-ubyte');
XTest = flatten(processImagesMNIST( './dataset/t10k-images.idx3-ubyte'), 28 * 28, 10000)';
YTest = processLabelsMNIST('./dataset/t10k-labels.idx1-ubyte');


input_size = 784;
hidden_size = 1000; 
output_size = 10;
iters_num = 300; 
train_set_size = 60000; 
test_set_size = 10000;

batch_size = 1000;
reg = 5e-4;
lr = 0.5;
std = 1;
loss_decay = 0.995;

W1 = std * randn(input_size, hidden_size);
b1 = zeros(1,hidden_size);
W2 = std * randn(hidden_size, output_size);
b2 = zeros(1,output_size);

loss_history = zeros(iters_num,1); 
train_acc_history = zeros(iters_num,1);
test_acc_history = zeros(iters_num,1);

fprintf("Start to train\n")
best_acc = 0.0;
for idx = 1:iters_num
    fprintf("Epoch %d:\n", idx);
    tic
    perm = randperm(train_set_size);
    X = XTrain(perm,:);
    Y = YTrain(perm);
    batch_len = ceil(train_set_size / batch_size);
    test_batch_len = ceil(test_set_size / batch_size);
    train_acc = 0.0;
    test_acc = 0.0;
    loss_avg = 0.0;

    for batch_idx = 1:batch_len
        % randomly split dataset into mini-batch
        X_batch = X((batch_idx -1) * batch_size + 1:batch_idx * batch_size, :);
        Y_batch = Y((batch_idx -1) * batch_size + 1:batch_idx * batch_size, :);
        % get loss
        [loss, grads_W1, grads_b1, grads_W2, grads_b2] = get_loss_and_grad(X_batch, Y_batch, W1, b1, W2, b2, reg, batch_size);
        loss_avg = loss_avg + extractdata(sum(loss));
        % upgrade gradients
        W1 = W1 - lr * grads_W1;
        b1 = b1 - lr * grads_b1;
        W2 = W2 - lr * grads_W2;
        b2 = b2 - lr * grads_b2;
        % get predict result
        result = extractdata(predict(X_batch, W1, b1, W2, b2));
        result = sum(result == Y_batch);
        train_acc = train_acc + result;
    end
    lr = lr * loss_decay;

    for batch_idx = 1:test_batch_len
        X_batch = XTest((batch_idx -1) * batch_size + 1 : batch_idx * batch_size, :);
        Y_batch = YTest((batch_idx -1) * batch_size + 1 : batch_idx * batch_size, :);
        result = extractdata(predict(X_batch, W1, b1, W2, b2));
        result = sum(result == Y_batch);
        test_acc = test_acc + result;
    end
    fprintf("train loss=%.4f, train acc=%.4f%%, test acc=%.4f%%\n",loss_avg / train_set_size, 100 * train_acc / train_set_size, 100 * test_acc / test_set_size);
    loss_history(idx) = loss_avg / train_set_size;
    train_acc_history(idx) =  train_acc / train_set_size;
    if (test_acc > best_acc)
        best_acc = test_acc;
        fprintf("Saving model to model.mat\n");
        save model W1 b1 W2 b2
    end
        
    test_acc_history(idx) =  test_acc / test_set_size;
    toc
end

figure()
plot(train_acc_history)
hold on
plot(test_acc_history)
legend train test




