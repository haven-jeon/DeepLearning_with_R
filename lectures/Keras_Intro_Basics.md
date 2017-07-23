Keras\_Intro\_Basics
================

### Keras

-   빠른 모형 프로토타이핑을 통한 연구 효율 증대
-   API사용이 쉽고 오사용으로 인한 시행착오를 줄여준다.
-   GPU, CPU 모두 사용 가능
-   다양한 backend 활용 가능

#### R interface to Keras

-   Python 버전의 Keras를 R에서 활용 가능하게 Wrapping한 패키지
-   R의 강점을 딥러닝을 하는데 활용 가능
-   간결한 API, 복잡한 import사용을 하지 않아도 된다.
-   reticulate 기반으로 동작해 keras의 최신 기능을 바로 활용 가능하다.

### Keras 설치

    devtools::install_github("rstudio/keras")
    library(keras)
    install_tensorflow()

### A first neural network in Keras

``` r
library(keras)
library(reticulate)
```

#### 데이터 로딩과 샘플 플로팅

``` r
batch_size <- 32
num_classes <- 10
epochs <- 10

# the data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
```

``` r
x_train <- array(as.numeric(x_train), dim = c(dim(x_train)[[1]], 784))
x_test <- array(as.numeric(x_test), dim = c(dim(x_test)[[1]], 784))

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)


print(sprintf("x_train dimension : %d, %d", dim(x_train)[1], dim(x_train)[2]))
```

    ## [1] "x_train dimension : 60000, 784"

``` r
print(sprintf("y_train dimension : %d, %d", dim(y_train)[1], dim(y_train)[2]))
```

    ## [1] "y_train dimension : 60000, 10"

``` r
plot_examples <- function(data, labels, model_predict){
    par(mfrow=c(2,4))
    for(i in 1:8){
        idx <- sample(seq(dim(data)[1]), 1)
        lab <- paste0("num: " , which.max(labels[idx,]) - 1 , "," , which.max(model_predict[idx,]) - 1)
        image(t(apply(array(data[idx,], dim = c(28,28)), 2, rev)),
              col=paste("gray",1:99,sep=""),main=lab)
    }
}

plot_examples(x_train, y_train, y_train)
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-4-1.png)

#### Building a first Keras model

-   간단한 3층의 앞먹임(feedforward) 네트워크
-   3 step:
    -   모델 빌드
        -   옵티마이저(optimizer)와 로스(loss)를 기반으로 모델 컴파일
            -   모형에 저장된 가중치와 바이어스에 영향 없이 learning rate와 optimizer 등의 설정을 바꿀 수 있다.
        -   `model %>% fit`을 호출해 학습을 시작한다.

``` r
# Keras는 두가지 모델링 API를 제공한다.  
#    1. Sequential - 복잡하지 않은 단순한 네트웍을 구성할때 
#    2. Functional - 복잡한 네트웍을 구성할때 

# input layer
inputs <- layer_input(shape = c(784))
 

# 모형 구조 구성 
# keras는 각 레이어를 순서에 맞게 구성하면 해당 입력 차원에 대한 명시적인 선언을 하지 않아도 된다. 
# 해당 레이어에서 출력되는 차원을 명시하고 activation 함수를 정의한다.  
predictions <- inputs %>%
  layer_dense(units = 128, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'sigmoid')

# 모형을 생성한다. 
model <- keras_model(inputs = inputs, outputs = predictions)

#모형 구조 출력 
summary(model)
```

    ## Model
    ## ___________________________________________________________________________
    ## Layer (type)                     Output Shape                  Param #     
    ## ===========================================================================
    ## input_1 (InputLayer)             (None, 784)                   0           
    ## ___________________________________________________________________________
    ## dense_1 (Dense)                  (None, 128)                   100480      
    ## ___________________________________________________________________________
    ## dense_2 (Dense)                  (None, 10)                    1290        
    ## ===========================================================================
    ## Total params: 101,770
    ## Trainable params: 101,770
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
#컴파일 과정을 통해 최적화 조건을 선언한다. 
model %>% compile(
  optimizer = sgd_lr,
  loss = 'mse',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2
  )
},type = 'stdout'))
```

    ## Epoch 1/10
    ## 2s - loss: 0.0863 - acc: 0.4224
    ## Epoch 2/10
    ## 1s - loss: 0.0601 - acc: 0.6751
    ## Epoch 3/10
    ## 1s - loss: 0.0498 - acc: 0.7624
    ## Epoch 4/10
    ## 1s - loss: 0.0430 - acc: 0.8078
    ## Epoch 5/10
    ## 1s - loss: 0.0383 - acc: 0.8328
    ## Epoch 6/10
    ## 1s - loss: 0.0349 - acc: 0.8481
    ## Epoch 7/10
    ## 1s - loss: 0.0323 - acc: 0.8602
    ## Epoch 8/10
    ## 1s - loss: 0.0302 - acc: 0.8682
    ## Epoch 9/10
    ## 1s - loss: 0.0286 - acc: 0.8749
    ## Epoch 10/10
    ## 1s - loss: 0.0272 - acc: 0.8802

``` r
#테스트셋 검증 
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
  
cat('Test loss:', score[[1]], '\n')
```

    ## Test loss: 0.02628412

``` r
cat('Test accuracy:', score[[2]], '\n')
```

    ## Test accuracy: 0.8824

``` r
#plot history of epoch
plot(history)
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-1.png)

#### 모델 재컴파일 예제

-   학습 중간에 옵티마이저와 로스함수를 수정할 수 있다.
-   이 기능은 모형의 업데이트가 크지 않을때 학습율을 조정하기 쉽게 한다.
-   여기서는 `validation_split`옵션을 이용해 학습 데이터의 일정 부분을 테스트 하기 위한 용도로 활용하고 이 결과를 학습이 진행되는 중간 중간 출력해준다.
    -   별도 셋을 구분하지 않아도 일정부분의 데이터를 기반으로 각 에폭 마지막에 loss와 metric을 계산해 줄력해준다.

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 128, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'sigmoid')


model <- keras_model(inputs = inputs, outputs = predictions)

print("Learning rate is 0.1")
```

    ## [1] "Learning rate is 0.1"

``` r
sgd_lr <- optimizer_sgd(lr=0.1)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'mse',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({ 
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_split= 0.2
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 1s - loss: 0.0502 - acc: 0.7232 - val_loss: 0.0305 - val_acc: 0.8641
    ## Epoch 2/10
    ## 1s - loss: 0.0273 - acc: 0.8720 - val_loss: 0.0221 - val_acc: 0.8970
    ## Epoch 3/10
    ## 1s - loss: 0.0218 - acc: 0.8932 - val_loss: 0.0187 - val_acc: 0.9080
    ## Epoch 4/10
    ## 1s - loss: 0.0188 - acc: 0.9057 - val_loss: 0.0168 - val_acc: 0.9151
    ## Epoch 5/10
    ## 1s - loss: 0.0169 - acc: 0.9135 - val_loss: 0.0159 - val_acc: 0.9171
    ## Epoch 6/10
    ## 1s - loss: 0.0158 - acc: 0.9188 - val_loss: 0.0147 - val_acc: 0.9237
    ## Epoch 7/10
    ## 1s - loss: 0.0148 - acc: 0.9224 - val_loss: 0.0139 - val_acc: 0.9277
    ## Epoch 8/10
    ## 1s - loss: 0.0141 - acc: 0.9245 - val_loss: 0.0134 - val_acc: 0.9297
    ## Epoch 9/10
    ## 1s - loss: 0.0134 - acc: 0.9288 - val_loss: 0.0129 - val_acc: 0.9320
    ## Epoch 10/10
    ## 1s - loss: 0.0129 - acc: 0.9306 - val_loss: 0.0126 - val_acc: 0.9325

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
print("Learning rate is 0.01")
```

    ## [1] "Learning rate is 0.01"

``` r
model %>% compile(
  optimizer = sgd_lr,
  loss = 'mse',
  metrics = c('accuracy')
)

#학습
cat(py_capture_output({ 
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_split= 0.2
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 1s - loss: 0.0121 - acc: 0.9354 - val_loss: 0.0120 - val_acc: 0.9347
    ## Epoch 2/10
    ## 1s - loss: 0.0117 - acc: 0.9378 - val_loss: 0.0118 - val_acc: 0.9347
    ## Epoch 3/10
    ## 1s - loss: 0.0115 - acc: 0.9390 - val_loss: 0.0117 - val_acc: 0.9362
    ## Epoch 4/10
    ## 1s - loss: 0.0113 - acc: 0.9402 - val_loss: 0.0116 - val_acc: 0.9369
    ## Epoch 5/10
    ## 1s - loss: 0.0111 - acc: 0.9413 - val_loss: 0.0116 - val_acc: 0.9357
    ## Epoch 6/10
    ## 1s - loss: 0.0110 - acc: 0.9418 - val_loss: 0.0116 - val_acc: 0.9370
    ## Epoch 7/10
    ## 1s - loss: 0.0109 - acc: 0.9425 - val_loss: 0.0115 - val_acc: 0.9377
    ## Epoch 8/10
    ## 1s - loss: 0.0107 - acc: 0.9437 - val_loss: 0.0115 - val_acc: 0.9371
    ## Epoch 9/10
    ## 1s - loss: 0.0106 - acc: 0.9442 - val_loss: 0.0115 - val_acc: 0.9369
    ## Epoch 10/10
    ## 1s - loss: 0.0106 - acc: 0.9447 - val_loss: 0.0114 - val_acc: 0.9373

``` r
sgd_lr <- optimizer_sgd(lr=0.001)
print("Learning rate is 0.001")
```

    ## [1] "Learning rate is 0.001"

``` r
model %>% compile(
  optimizer = sgd_lr,
  loss = 'mse',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_split= 0.2
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 1s - loss: 0.0104 - acc: 0.9459 - val_loss: 0.0114 - val_acc: 0.9374
    ## Epoch 2/10
    ## 1s - loss: 0.0104 - acc: 0.9460 - val_loss: 0.0114 - val_acc: 0.9373
    ## Epoch 3/10
    ## 1s - loss: 0.0104 - acc: 0.9462 - val_loss: 0.0114 - val_acc: 0.9374
    ## Epoch 4/10
    ## 1s - loss: 0.0103 - acc: 0.9462 - val_loss: 0.0114 - val_acc: 0.9374
    ## Epoch 5/10
    ## 1s - loss: 0.0103 - acc: 0.9464 - val_loss: 0.0114 - val_acc: 0.9377
    ## Epoch 6/10
    ## 1s - loss: 0.0103 - acc: 0.9465 - val_loss: 0.0114 - val_acc: 0.9377
    ## Epoch 7/10
    ## 1s - loss: 0.0103 - acc: 0.9465 - val_loss: 0.0114 - val_acc: 0.9375
    ## Epoch 8/10
    ## 1s - loss: 0.0103 - acc: 0.9467 - val_loss: 0.0114 - val_acc: 0.9373
    ## Epoch 9/10
    ## 1s - loss: 0.0103 - acc: 0.9466 - val_loss: 0.0114 - val_acc: 0.9371
    ## Epoch 10/10
    ## 1s - loss: 0.0103 - acc: 0.9467 - val_loss: 0.0114 - val_acc: 0.9374

#### 정확도(accuracy)출력을 위한 함수 작성

``` r
accuracy <- function(test_x, test_y, model){
  result <- predict(model,test_x)
  num_correct <- apply(result, 1, which.max) == apply(test_y, 1, which.max)
  accuracy <- sum(num_correct) / dim(result)[1]
  print(sprintf("Accuracy on data is: %f",accuracy * 100))
}
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 93.710000"

### 오분류, 정분류 시각화를 통한 Error Analysis

``` r
get_correct_and_incorrect <- function(model, test_x, test_y){
  result <- predict(model,test_x)
  correct_indices <- apply(result, 1, which.max) == apply(test_y, 1, which.max)
  test_x_correct <- test_x[correct_indices,]
  test_y_correct <- test_y[correct_indices,]
  predict_test_y_correct <- result[correct_indices,]
  incorrect_indices <- apply(result, 1, which.max) != apply(test_y, 1, which.max)
  test_x_incorrect <- test_x[incorrect_indices,]
  test_y_incorrect <- test_y[incorrect_indices,]
  predict_test_y_incorrect <- result[incorrect_indices,]
  return(list(test_x_correct, test_y_correct, test_x_incorrect, test_y_incorrect, predict_test_y_correct, predict_test_y_incorrect))
}

predit_res <- get_correct_and_incorrect(model, x_test, y_test)
```

``` r
print(dim(predit_res[[1]]))
```

    ## [1] 9371  784

``` r
plot_examples(predit_res[[1]], predit_res[[2]], predit_res[[5]])
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-9-1.png)

``` r
print(dim(predit_res[[3]]))
```

    ## [1] 629 784

``` r
plot_examples(predit_res[[3]], predit_res[[4]], predit_res[[6]])
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-10-1.png)

#### MSE vs. categorical crossentropy loss functions

-   카테고리컬 크로스 엔트로피 loss는 MSE보다 더 빠른 수렴 효과를 보인다.
-   Softmax output은 10개중에 하나의 클래스를 선택하는 MNIST문제에 적합하다.
    -   한 노드의 확률이 높아지면 다른 노드의 확률들은 낮아져야 된다.

``` r
# Softmax output layer, mse
print("Quadratic (MSE)")
```

    ## [1] "Quadratic (MSE)"

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 128, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)


sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'mse',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 10,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/mse")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 1s - loss: 0.0937 - acc: 0.1788 - val_loss: 0.0909 - val_acc: 0.2136
    ## Epoch 2/10
    ## 1s - loss: 0.0891 - acc: 0.2329 - val_loss: 0.0873 - val_acc: 0.2562
    ## Epoch 3/10
    ## 1s - loss: 0.0858 - acc: 0.2681 - val_loss: 0.0844 - val_acc: 0.2916
    ## Epoch 4/10
    ## 1s - loss: 0.0831 - acc: 0.2979 - val_loss: 0.0818 - val_acc: 0.3212
    ## Epoch 5/10
    ## 1s - loss: 0.0807 - acc: 0.3233 - val_loss: 0.0794 - val_acc: 0.3458
    ## Epoch 6/10
    ## 1s - loss: 0.0784 - acc: 0.3521 - val_loss: 0.0772 - val_acc: 0.3708
    ## Epoch 7/10
    ## 1s - loss: 0.0763 - acc: 0.3784 - val_loss: 0.0752 - val_acc: 0.3957
    ## Epoch 8/10
    ## 1s - loss: 0.0743 - acc: 0.4053 - val_loss: 0.0732 - val_acc: 0.4259
    ## Epoch 9/10
    ## 1s - loss: 0.0724 - acc: 0.4312 - val_loss: 0.0713 - val_acc: 0.4525
    ## Epoch 10/10
    ## 1s - loss: 0.0705 - acc: 0.4579 - val_loss: 0.0694 - val_acc: 0.4798

``` r
# Softmax output layer, categorical crossentropy
print("Categorical cross entropy")
```

    ## [1] "Categorical cross entropy"

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 128, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)


sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 10,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/categorical_crossentropy")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 1s - loss: 1.3991 - acc: 0.6027 - val_loss: 0.9160 - val_acc: 0.7977
    ## Epoch 2/10
    ## 1s - loss: 0.7934 - acc: 0.8141 - val_loss: 0.6550 - val_acc: 0.8522
    ## Epoch 3/10
    ## 1s - loss: 0.6214 - acc: 0.8527 - val_loss: 0.5449 - val_acc: 0.8747
    ## Epoch 4/10
    ## 1s - loss: 0.5338 - acc: 0.8724 - val_loss: 0.4829 - val_acc: 0.8843
    ## Epoch 5/10
    ## 1s - loss: 0.4786 - acc: 0.8829 - val_loss: 0.4379 - val_acc: 0.8944
    ## Epoch 6/10
    ## 1s - loss: 0.4376 - acc: 0.8916 - val_loss: 0.4086 - val_acc: 0.8990
    ## Epoch 7/10
    ## 1s - loss: 0.4070 - acc: 0.8977 - val_loss: 0.3817 - val_acc: 0.9056
    ## Epoch 8/10
    ## 1s - loss: 0.3837 - acc: 0.9031 - val_loss: 0.3630 - val_acc: 0.9063
    ## Epoch 9/10
    ## 1s - loss: 0.3630 - acc: 0.9074 - val_loss: 0.3490 - val_acc: 0.9117
    ## Epoch 10/10
    ## 1s - loss: 0.3470 - acc: 0.9102 - val_loss: 0.3343 - val_acc: 0.9133

#### ReLU vs. Sigmoid

##### ReLU

-   학습을 위해 다소 적은 learning rate 필요
-   깊이가 얕은 네트웍에서는 Sigmoid보다 성능이 덜 나오는 경향이 있다.

``` r
# Relu hidden layer, 3 layer network
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)


sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 10,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/relu")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 2s - loss: 9.4006 - acc: 0.4118 - val_loss: 7.5428 - val_acc: 0.5289
    ## Epoch 2/10
    ## 2s - loss: 7.2400 - acc: 0.5473 - val_loss: 5.8258 - val_acc: 0.6345
    ## Epoch 3/10
    ## 2s - loss: 5.7137 - acc: 0.6409 - val_loss: 4.3314 - val_acc: 0.7271
    ## Epoch 4/10
    ## 2s - loss: 4.5613 - acc: 0.7123 - val_loss: 4.2443 - val_acc: 0.7336
    ## Epoch 5/10
    ## 2s - loss: 4.3895 - acc: 0.7237 - val_loss: 4.1289 - val_acc: 0.7402
    ## Epoch 6/10
    ## 2s - loss: 3.9588 - acc: 0.7499 - val_loss: 2.9398 - val_acc: 0.8124
    ## Epoch 7/10
    ## 2s - loss: 2.9228 - acc: 0.8134 - val_loss: 2.8377 - val_acc: 0.8182
    ## Epoch 8/10
    ## 2s - loss: 2.7535 - acc: 0.8252 - val_loss: 2.6413 - val_acc: 0.8323
    ## Epoch 9/10
    ## 2s - loss: 2.7020 - acc: 0.8282 - val_loss: 2.5909 - val_acc: 0.8355
    ## Epoch 10/10
    ## 2s - loss: 2.5889 - acc: 0.8356 - val_loss: 2.4861 - val_acc: 0.8412

``` r
# Sigmoid hidden layer, 3 layer network
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 128, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)

sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 10,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/sigmoid")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 1s - loss: 1.4957 - acc: 0.5850 - val_loss: 0.9559 - val_acc: 0.7969
    ## Epoch 2/10
    ## 1s - loss: 0.8395 - acc: 0.8107 - val_loss: 0.6872 - val_acc: 0.8483
    ## Epoch 3/10
    ## 1s - loss: 0.6568 - acc: 0.8467 - val_loss: 0.5664 - val_acc: 0.8715
    ## Epoch 4/10
    ## 1s - loss: 0.5600 - acc: 0.8654 - val_loss: 0.4980 - val_acc: 0.8840
    ## Epoch 5/10
    ## 1s - loss: 0.4990 - acc: 0.8776 - val_loss: 0.4529 - val_acc: 0.8922
    ## Epoch 6/10
    ## 1s - loss: 0.4562 - acc: 0.8872 - val_loss: 0.4203 - val_acc: 0.8967
    ## Epoch 7/10
    ## 1s - loss: 0.4235 - acc: 0.8942 - val_loss: 0.3950 - val_acc: 0.9006
    ## Epoch 8/10
    ## 1s - loss: 0.3983 - acc: 0.9000 - val_loss: 0.3763 - val_acc: 0.9054
    ## Epoch 9/10
    ## 1s - loss: 0.3778 - acc: 0.9041 - val_loss: 0.3603 - val_acc: 0.9057
    ## Epoch 10/10
    ## 1s - loss: 0.3601 - acc: 0.9085 - val_loss: 0.3478 - val_acc: 0.9090

#### `Relu` 에게는 깊은 네트워크가 적합하다.

-   깊은 네트워크일 수록 예측 능력이 뛰어나다.
-   `Relu`는 깊은 네트워크일 수록 잘 동작하는데, 이는 positive input에 대해서 어떻게든지 gradient를 계산해서 가중치를 업데이트 하기 때문이다.

``` r
# Relu hidden layer, 6 layer network
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)


sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 10,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/relu_6layer")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 2s - loss: 2.4635 - acc: 0.8063 - val_loss: 0.5915 - val_acc: 0.9202
    ## Epoch 2/10
    ## 2s - loss: 0.4042 - acc: 0.9295 - val_loss: 0.2952 - val_acc: 0.9348
    ## Epoch 3/10
    ## 2s - loss: 0.1934 - acc: 0.9539 - val_loss: 0.2483 - val_acc: 0.9431
    ## Epoch 4/10
    ## 2s - loss: 0.1263 - acc: 0.9672 - val_loss: 0.2214 - val_acc: 0.9491
    ## Epoch 5/10
    ## 2s - loss: 0.0834 - acc: 0.9785 - val_loss: 0.2068 - val_acc: 0.9496
    ## Epoch 6/10
    ## 2s - loss: 0.0591 - acc: 0.9855 - val_loss: 0.2071 - val_acc: 0.9524
    ## Epoch 7/10
    ## 2s - loss: 0.0416 - acc: 0.9910 - val_loss: 0.2102 - val_acc: 0.9534
    ## Epoch 8/10
    ## 2s - loss: 0.0314 - acc: 0.9941 - val_loss: 0.2065 - val_acc: 0.9538
    ## Epoch 9/10
    ## 2s - loss: 0.0251 - acc: 0.9961 - val_loss: 0.2053 - val_acc: 0.9559
    ## Epoch 10/10
    ## 2s - loss: 0.0202 - acc: 0.9972 - val_loss: 0.2107 - val_acc: 0.9563

``` r
# Relu hidden layer, 6 layer network
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 512, activation = 'sigmoid') %>% 
  layer_dense(units = 256, activation = 'sigmoid') %>% 
  layer_dense(units = 128, activation = 'sigmoid') %>% 
  layer_dense(units = 64, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)


sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 10,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/sigmoid_6layer")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/10
    ## 2s - loss: 2.3227 - acc: 0.1066 - val_loss: 2.3018 - val_acc: 0.1060
    ## Epoch 2/10
    ## 2s - loss: 2.2990 - acc: 0.1140 - val_loss: 2.2985 - val_acc: 0.1060
    ## Epoch 3/10
    ## 2s - loss: 2.2958 - acc: 0.1140 - val_loss: 2.2953 - val_acc: 0.1060
    ## Epoch 4/10
    ## 2s - loss: 2.2926 - acc: 0.1140 - val_loss: 2.2921 - val_acc: 0.1060
    ## Epoch 5/10
    ## 2s - loss: 2.2893 - acc: 0.1140 - val_loss: 2.2890 - val_acc: 0.1060
    ## Epoch 6/10
    ## 2s - loss: 2.2860 - acc: 0.1140 - val_loss: 2.2848 - val_acc: 0.1062
    ## Epoch 7/10
    ## 2s - loss: 2.2824 - acc: 0.1144 - val_loss: 2.2813 - val_acc: 0.1060
    ## Epoch 8/10
    ## 2s - loss: 2.2786 - acc: 0.1141 - val_loss: 2.2774 - val_acc: 0.1060
    ## Epoch 9/10
    ## 2s - loss: 2.2746 - acc: 0.1152 - val_loss: 2.2731 - val_acc: 0.1060
    ## Epoch 10/10
    ## 2s - loss: 2.2700 - acc: 0.1170 - val_loss: 2.2681 - val_acc: 0.1438

#### 마지막 모형 학습

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')


model <- keras_model(inputs = inputs, outputs = predictions)

summary(model)
```

    ## Model
    ## ___________________________________________________________________________
    ## Layer (type)                     Output Shape                  Param #     
    ## ===========================================================================
    ## input_9 (InputLayer)             (None, 784)                   0           
    ## ___________________________________________________________________________
    ## dense_23 (Dense)                 (None, 512)                   401920      
    ## ___________________________________________________________________________
    ## dense_24 (Dense)                 (None, 256)                   131328      
    ## ___________________________________________________________________________
    ## dense_25 (Dense)                 (None, 128)                   32896       
    ## ___________________________________________________________________________
    ## dense_26 (Dense)                 (None, 64)                    8256        
    ## ___________________________________________________________________________
    ## dense_27 (Dense)                 (None, 10)                    650         
    ## ===========================================================================
    ## Total params: 575,050
    ## Trainable params: 575,050
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.001)

model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

cat(py_capture_output({
  history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 15,
    verbose = 2,
    validation_split= 0.2,
    callbacks = callback_tensorboard(log_dir = "logs/final")
)},type = 'stdout'))
```

    ## Train on 48000 samples, validate on 12000 samples
    ## Epoch 1/15
    ## 2s - loss: 7.0749 - acc: 0.5510 - val_loss: 6.6655 - val_acc: 0.5792
    ## Epoch 2/15
    ## 2s - loss: 5.5045 - acc: 0.6396 - val_loss: 2.3799 - val_acc: 0.8057
    ## Epoch 3/15
    ## 2s - loss: 0.6345 - acc: 0.9003 - val_loss: 0.2729 - val_acc: 0.9283
    ## Epoch 4/15
    ## 2s - loss: 0.1955 - acc: 0.9478 - val_loss: 0.2349 - val_acc: 0.9370
    ## Epoch 5/15
    ## 2s - loss: 0.1305 - acc: 0.9635 - val_loss: 0.2161 - val_acc: 0.9423
    ## Epoch 6/15
    ## 2s - loss: 0.0915 - acc: 0.9734 - val_loss: 0.2106 - val_acc: 0.9465
    ## Epoch 7/15
    ## 2s - loss: 0.0682 - acc: 0.9809 - val_loss: 0.2017 - val_acc: 0.9489
    ## Epoch 8/15
    ## 2s - loss: 0.0515 - acc: 0.9856 - val_loss: 0.2034 - val_acc: 0.9499
    ## Epoch 9/15
    ## 2s - loss: 0.0405 - acc: 0.9891 - val_loss: 0.1985 - val_acc: 0.9533
    ## Epoch 10/15
    ## 2s - loss: 0.0315 - acc: 0.9927 - val_loss: 0.1971 - val_acc: 0.9547
    ## Epoch 11/15
    ## 2s - loss: 0.0249 - acc: 0.9951 - val_loss: 0.2005 - val_acc: 0.9555
    ## Epoch 12/15
    ## 2s - loss: 0.0206 - acc: 0.9961 - val_loss: 0.1996 - val_acc: 0.9560
    ## Epoch 13/15
    ## 2s - loss: 0.0168 - acc: 0.9975 - val_loss: 0.2002 - val_acc: 0.9571
    ## Epoch 14/15
    ## 2s - loss: 0.0143 - acc: 0.9982 - val_loss: 0.2033 - val_acc: 0.9569
    ## Epoch 15/15
    ## 2s - loss: 0.0122 - acc: 0.9988 - val_loss: 0.2056 - val_acc: 0.9581

#### 학습 데이터 기반 모형 성능

``` r
accuracy(x_train, y_train, model)
```

    ## [1] "Accuracy on data is: 99.091667"

``` r
final_res <- get_correct_and_incorrect(model, x_train, y_train)
```

``` r
print(dim(final_res[[1]]))
```

    ## [1] 59455   784

``` r
plot_examples(final_res[[1]], final_res[[2]], final_res[[5]])
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-19-1.png)

``` r
print(dim(final_res[[3]]))
```

    ## [1] 545 784

``` r
plot_examples(final_res[[3]], final_res[[4]], final_res[[6]])
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-20-1.png)

#### 테스트 데이터 기반 모형 성능

``` r
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 95.590000"

``` r
final_test_res <- get_correct_and_incorrect(model, x_test, y_test)
```

``` r
print(dim(final_test_res[[1]]))
```

    ## [1] 9559  784

``` r
plot_examples(final_test_res[[1]], final_test_res[[2]], final_test_res[[5]])
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-23-1.png)

``` r
print(dim(final_test_res[[3]]))
```

    ## [1] 441 784

``` r
plot_examples(final_test_res[[3]], final_test_res[[4]], final_test_res[[6]])
```

![](Keras_Intro_Basics_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-24-1.png)

``` r
#텐서보드 띄워 보기 
tensorboard(log_dir = 'lectures/logs/',host = '0.0.0.0', port = 8002)
```

![tensorboard](TensorBoard.png)
