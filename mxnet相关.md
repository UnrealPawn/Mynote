## Use GPUs
在Mxnet里面使用GPU只需2步
1. 把输入数据放入GPU
2. 把网络参数放入GPU

we only need to copy/move the input data and parameters to the GPU

---

-  安装pip install mxnet-cu90 

（replace `cu90` according to your CUDA version.查看cuda版本 nvcc -version）

---
- 单个GPU训练
```
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10))
把网络参数放入GPU:        
net.load_parameters('net.params', ctx=gpu(0))
```

```
把输入数据放入GPU:
x = nd.random.uniform(shape=(1,1,28,28), ctx=gpu(0))
net(x)
```


---
- 多GPU训练
```
# Diff 1: Use two GPUs for training.
devices = [gpu(0), gpu(1)]
# Diff 2: reinitialize the parameters and place them on multiple GPUs
net.collect_params().initialize(force_reinit=True, ctx=devices)
# Loss and trainer are the same as before
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
for epoch in range(10):
    train_loss = 0.
    tic = time.time()
    for data, label in train_data:
        # Diff 3: split batch and load into corresponding devices
        data_list = gluon.utils.split_and_load(data, devices)
        label_list = gluon.utils.split_and_load(label, devices)
        # Diff 4: run forward and backward on each devices.
        # MXNet will automatically run them in parallel
        with autograd.record():
            losses = [softmax_cross_entropy(net(X), y)
                      for X, y in zip(data_list, label_list)]
        for l in losses:
            l.backward()
        trainer.step(batch_size)
        # Diff 5: sum losses over all devices
        train_loss += sum([l.sum().asscalar() for l in losses])
    print("Epoch %d: loss %.3f, in %.1f sec" % (
        epoch, train_loss/len(train_data)/batch_size, time.time()-tic))
```
