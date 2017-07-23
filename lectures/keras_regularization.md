Keras Regularization
================

#### 패키지 로딩

``` r
library(keras)
library(reticulate)
```

#### 데이터 셋 및 시각화 함수 정의

``` r
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array(as.numeric(x_train), dim = c(dim(x_train)[[1]], 784))
x_test <- array(as.numeric(x_test), dim = c(dim(x_test)[[1]], 784))

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

x_train_short <- x_train[1:5000,]
y_train_short <- y_train[1:5000,]

print(dim(x_train_short))
```

    ## [1] 5000  784

``` r
print(dim(y_train_short))
```

    ## [1] 5000   10

#### 예제 시각화

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

![](keras_regularization_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-3-1.png)

``` r
plot_examples(x_test, y_test, y_test)
```

![](keras_regularization_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-4-1.png)

#### 동기 부여를 위해 적은 데이터로 너무 많은 노드를 추가해 보자!

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 1024, activation = 'sigmoid') %>% 
  layer_dense(units = 10, activation = 'softmax')

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
    ## dense_1 (Dense)                  (None, 1024)                  803840      
    ## ___________________________________________________________________________
    ## dense_2 (Dense)                  (None, 10)                    10250       
    ## ===========================================================================
    ## Total params: 814,090
    ## Trainable params: 814,090
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
#컴파일 과정을 통해 최적화 조건을 선언한다. 
model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train_short, y_train_short,
    batch_size = 32,
    epochs = 30,
    verbose = 2,
    validation_split=0.9, 
    callback_tensorboard(log_dir = "logs2/first")
  )
},type = 'stdout'))
```

    ## Train on 499 samples, validate on 4501 samples
    ## Epoch 1/30
    ## 0s - loss: 1.8857 - acc: 0.3687 - val_loss: 1.6363 - val_acc: 0.4650
    ## Epoch 2/30
    ## 0s - loss: 1.0488 - acc: 0.7735 - val_loss: 1.2258 - val_acc: 0.6685
    ## Epoch 3/30
    ## 0s - loss: 0.7082 - acc: 0.8938 - val_loss: 1.0795 - val_acc: 0.6992
    ## Epoch 4/30
    ## 0s - loss: 0.5126 - acc: 0.9339 - val_loss: 0.9963 - val_acc: 0.7207
    ## Epoch 5/30
    ## 0s - loss: 0.4103 - acc: 0.9619 - val_loss: 0.9356 - val_acc: 0.7434
    ## Epoch 6/30
    ## 0s - loss: 0.3281 - acc: 0.9780 - val_loss: 0.8964 - val_acc: 0.7518
    ## Epoch 7/30
    ## 0s - loss: 0.2795 - acc: 0.9880 - val_loss: 0.8658 - val_acc: 0.7549
    ## Epoch 8/30
    ## 0s - loss: 0.2421 - acc: 0.9900 - val_loss: 0.8435 - val_acc: 0.7547
    ## Epoch 9/30
    ## 0s - loss: 0.2177 - acc: 0.9920 - val_loss: 0.8281 - val_acc: 0.7581
    ## Epoch 10/30
    ## 0s - loss: 0.1976 - acc: 0.9960 - val_loss: 0.8105 - val_acc: 0.7638
    ## Epoch 11/30
    ## 0s - loss: 0.1823 - acc: 0.9960 - val_loss: 0.8037 - val_acc: 0.7661
    ## Epoch 12/30
    ## 0s - loss: 0.1700 - acc: 0.9960 - val_loss: 0.7894 - val_acc: 0.7694
    ## Epoch 13/30
    ## 0s - loss: 0.1598 - acc: 0.9960 - val_loss: 0.7797 - val_acc: 0.7705
    ## Epoch 14/30
    ## 0s - loss: 0.1495 - acc: 0.9960 - val_loss: 0.7731 - val_acc: 0.7729
    ## Epoch 15/30
    ## 0s - loss: 0.1423 - acc: 0.9980 - val_loss: 0.7645 - val_acc: 0.7738
    ## Epoch 16/30
    ## 0s - loss: 0.1363 - acc: 0.9980 - val_loss: 0.7596 - val_acc: 0.7718
    ## Epoch 17/30
    ## 0s - loss: 0.1300 - acc: 0.9980 - val_loss: 0.7535 - val_acc: 0.7732
    ## Epoch 18/30
    ## 0s - loss: 0.1248 - acc: 0.9980 - val_loss: 0.7458 - val_acc: 0.7763
    ## Epoch 19/30
    ## 0s - loss: 0.1200 - acc: 0.9980 - val_loss: 0.7403 - val_acc: 0.7767
    ## Epoch 20/30
    ## 0s - loss: 0.1156 - acc: 0.9980 - val_loss: 0.7349 - val_acc: 0.7787
    ## Epoch 21/30
    ## 0s - loss: 0.1114 - acc: 0.9980 - val_loss: 0.7296 - val_acc: 0.7780
    ## Epoch 22/30
    ## 0s - loss: 0.1075 - acc: 0.9980 - val_loss: 0.7253 - val_acc: 0.7812
    ## Epoch 23/30
    ## 0s - loss: 0.1037 - acc: 0.9980 - val_loss: 0.7213 - val_acc: 0.7823
    ## Epoch 24/30
    ## 0s - loss: 0.1007 - acc: 0.9980 - val_loss: 0.7197 - val_acc: 0.7812
    ## Epoch 25/30
    ## 0s - loss: 0.0973 - acc: 1.0000 - val_loss: 0.7155 - val_acc: 0.7825
    ## Epoch 26/30
    ## 0s - loss: 0.0943 - acc: 1.0000 - val_loss: 0.7095 - val_acc: 0.7858
    ## Epoch 27/30
    ## 0s - loss: 0.0917 - acc: 1.0000 - val_loss: 0.7063 - val_acc: 0.7843
    ## Epoch 28/30
    ## 0s - loss: 0.0892 - acc: 1.0000 - val_loss: 0.7053 - val_acc: 0.7825
    ## Epoch 29/30
    ## 0s - loss: 0.0867 - acc: 1.0000 - val_loss: 0.7003 - val_acc: 0.7863
    ## Epoch 30/30
    ## 0s - loss: 0.0842 - acc: 1.0000 - val_loss: 0.6974 - val_acc: 0.7867

#### Helper functions

``` r
accuracy <- function(test_x, test_y, model){
  result <- predict(model,test_x)
  num_correct <- apply(result, 1, which.max) == apply(test_y, 1, which.max)
  accuracy <- sum(num_correct) / dim(result)[1]
  print(sprintf("Accuracy on data is: %f",accuracy * 100))
}

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
```

``` r
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 78.930000"

#### L1 and L2 weight regularization

-   L2 가중치 정규화
    -   정규화 파라메터 수정을 하게 되면 어떠한 현상이 일어나는가?
    -   loss에는 어떠한 현상이 발생되는가?

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 1024, activation = 'sigmoid',kernel_regularizer=regularizer_l2(0.1)) %>% 
  layer_dense(units = 10, activation = 'softmax')

# 모형을 생성한다. 
model <- keras_model(inputs = inputs, outputs = predictions)

#모형 구조 출력 
summary(model)
```

    ## Model
    ## ___________________________________________________________________________
    ## Layer (type)                     Output Shape                  Param #     
    ## ===========================================================================
    ## input_2 (InputLayer)             (None, 784)                   0           
    ## ___________________________________________________________________________
    ## dense_3 (Dense)                  (None, 1024)                  803840      
    ## ___________________________________________________________________________
    ## dense_4 (Dense)                  (None, 10)                    10250       
    ## ===========================================================================
    ## Total params: 814,090
    ## Trainable params: 814,090
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
#컴파일 과정을 통해 최적화 조건을 선언한다. 
model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train_short, y_train_short,
    batch_size = 32,
    epochs = 30,
    verbose = 2,
    validation_split=0.9,
    callback_tensorboard(log_dir = "logs2/l2")
  )
},type = 'stdout'))
```

    ## Train on 499 samples, validate on 4501 samples
    ## Epoch 1/30
    ## 0s - loss: 88.2776 - acc: 0.3768 - val_loss: 84.9200 - val_acc: 0.5525
    ## Epoch 2/30
    ## 0s - loss: 82.0429 - acc: 0.7715 - val_loss: 79.3891 - val_acc: 0.7003
    ## Epoch 3/30
    ## 0s - loss: 76.6679 - acc: 0.8657 - val_loss: 74.4402 - val_acc: 0.6950
    ## Epoch 4/30
    ## 0s - loss: 71.7889 - acc: 0.9299 - val_loss: 69.7710 - val_acc: 0.7436
    ## Epoch 5/30
    ## 0s - loss: 67.2526 - acc: 0.9619 - val_loss: 65.4530 - val_acc: 0.7449
    ## Epoch 6/30
    ## 0s - loss: 63.0275 - acc: 0.9780 - val_loss: 61.3977 - val_acc: 0.7698
    ## Epoch 7/30
    ## 0s - loss: 59.0804 - acc: 0.9860 - val_loss: 57.6100 - val_acc: 0.7709
    ## Epoch 8/30
    ## 0s - loss: 55.3962 - acc: 0.9960 - val_loss: 54.0568 - val_acc: 0.7823
    ## Epoch 9/30
    ## 0s - loss: 51.9462 - acc: 0.9980 - val_loss: 50.7354 - val_acc: 0.7820
    ## Epoch 10/30
    ## 0s - loss: 48.7124 - acc: 0.9980 - val_loss: 47.6277 - val_acc: 0.7785
    ## Epoch 11/30
    ## 0s - loss: 45.6844 - acc: 1.0000 - val_loss: 44.7058 - val_acc: 0.7825
    ## Epoch 12/30
    ## 0s - loss: 42.8471 - acc: 1.0000 - val_loss: 41.9648 - val_acc: 0.7845
    ## Epoch 13/30
    ## 0s - loss: 40.1863 - acc: 1.0000 - val_loss: 39.4013 - val_acc: 0.7845
    ## Epoch 14/30
    ## 0s - loss: 37.6923 - acc: 1.0000 - val_loss: 36.9886 - val_acc: 0.7874
    ## Epoch 15/30
    ## 0s - loss: 35.3527 - acc: 1.0000 - val_loss: 34.7321 - val_acc: 0.7876
    ## Epoch 16/30
    ## 0s - loss: 33.1600 - acc: 1.0000 - val_loss: 32.6126 - val_acc: 0.7914
    ## Epoch 17/30
    ## 0s - loss: 31.1029 - acc: 1.0000 - val_loss: 30.6239 - val_acc: 0.7934
    ## Epoch 18/30
    ## 0s - loss: 29.1733 - acc: 1.0000 - val_loss: 28.7632 - val_acc: 0.7918
    ## Epoch 19/30
    ## 0s - loss: 27.3642 - acc: 1.0000 - val_loss: 27.0142 - val_acc: 0.7945
    ## Epoch 20/30
    ## 0s - loss: 25.6671 - acc: 1.0000 - val_loss: 25.3731 - val_acc: 0.7960
    ## Epoch 21/30
    ## 0s - loss: 24.0751 - acc: 1.0000 - val_loss: 23.8334 - val_acc: 0.7963
    ## Epoch 22/30
    ## 0s - loss: 22.5828 - acc: 1.0000 - val_loss: 22.3916 - val_acc: 0.7974
    ## Epoch 23/30
    ## 0s - loss: 21.1826 - acc: 1.0000 - val_loss: 21.0368 - val_acc: 0.7976
    ## Epoch 24/30
    ## 0s - loss: 19.8692 - acc: 1.0000 - val_loss: 19.7682 - val_acc: 0.7994
    ## Epoch 25/30
    ## 0s - loss: 18.6376 - acc: 1.0000 - val_loss: 18.5747 - val_acc: 0.7987
    ## Epoch 26/30
    ## 0s - loss: 17.4823 - acc: 1.0000 - val_loss: 17.4576 - val_acc: 0.7994
    ## Epoch 27/30
    ## 0s - loss: 16.3987 - acc: 1.0000 - val_loss: 16.4082 - val_acc: 0.8025
    ## Epoch 28/30
    ## 0s - loss: 15.3821 - acc: 1.0000 - val_loss: 15.4258 - val_acc: 0.8012
    ## Epoch 29/30
    ## 0s - loss: 14.4286 - acc: 1.0000 - val_loss: 14.5043 - val_acc: 0.8016
    ## Epoch 30/30
    ## 0s - loss: 13.5344 - acc: 1.0000 - val_loss: 13.6386 - val_acc: 0.8020

``` r
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 80.310000"

#### L1 weight regularization

-   정규화 파라메터 수정을 하게 되면 어떠한 현상이 일어나는가?

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 1024, activation = 'sigmoid',kernel_regularizer=regularizer_l1(0.1)) %>% 
  layer_dense(units = 10, activation = 'softmax')

# 모형을 생성한다. 
model <- keras_model(inputs = inputs, outputs = predictions)

#모형 구조 출력 
summary(model)
```

    ## Model
    ## ___________________________________________________________________________
    ## Layer (type)                     Output Shape                  Param #     
    ## ===========================================================================
    ## input_3 (InputLayer)             (None, 784)                   0           
    ## ___________________________________________________________________________
    ## dense_5 (Dense)                  (None, 1024)                  803840      
    ## ___________________________________________________________________________
    ## dense_6 (Dense)                  (None, 10)                    10250       
    ## ===========================================================================
    ## Total params: 814,090
    ## Trainable params: 814,090
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
#컴파일 과정을 통해 최적화 조건을 선언한다. 
model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train_short, y_train_short,
    batch_size = 32,
    epochs = 30,
    verbose = 2,
    validation_split=0.9,
    callback_tensorboard(log_dir = "logs2/l1")
  )
},type = 'stdout'))
```

    ## Train on 499 samples, validate on 4501 samples
    ## Epoch 1/30
    ## 0s - loss: 1788.3342 - acc: 0.3828 - val_loss: 1222.8736 - val_acc: 0.5685
    ## Epoch 2/30
    ## 0s - loss: 854.7837 - acc: 0.7776 - val_loss: 484.6151 - val_acc: 0.4817
    ## Epoch 3/30
    ## 0s - loss: 280.8579 - acc: 0.6513 - val_loss: 107.8897 - val_acc: 0.4392
    ## Epoch 4/30
    ## 0s - loss: 70.7797 - acc: 0.3547 - val_loss: 66.7029 - val_acc: 0.0906
    ## Epoch 5/30
    ## 0s - loss: 62.8874 - acc: 0.2204 - val_loss: 68.6644 - val_acc: 0.1560
    ## Epoch 6/30
    ## 0s - loss: 63.7823 - acc: 0.2004 - val_loss: 61.8840 - val_acc: 0.2384
    ## Epoch 7/30
    ## 0s - loss: 61.7316 - acc: 0.2104 - val_loss: 60.6562 - val_acc: 0.3510
    ## Epoch 8/30
    ## 0s - loss: 61.0315 - acc: 0.2685 - val_loss: 59.0955 - val_acc: 0.3064
    ## Epoch 9/30
    ## 0s - loss: 60.9942 - acc: 0.2545 - val_loss: 61.5167 - val_acc: 0.2353
    ## Epoch 10/30
    ## 0s - loss: 62.8333 - acc: 0.2224 - val_loss: 59.9431 - val_acc: 0.0984
    ## Epoch 11/30
    ## 0s - loss: 62.8444 - acc: 0.2305 - val_loss: 65.0958 - val_acc: 0.1626
    ## Epoch 12/30
    ## 0s - loss: 63.5020 - acc: 0.1844 - val_loss: 60.8161 - val_acc: 0.1126
    ## Epoch 13/30
    ## 0s - loss: 60.3590 - acc: 0.2405 - val_loss: 59.4412 - val_acc: 0.2899
    ## Epoch 14/30
    ## 0s - loss: 61.2889 - acc: 0.2285 - val_loss: 64.5462 - val_acc: 0.1740
    ## Epoch 15/30
    ## 0s - loss: 60.3224 - acc: 0.2265 - val_loss: 63.4935 - val_acc: 0.2293
    ## Epoch 16/30
    ## 0s - loss: 60.1980 - acc: 0.2385 - val_loss: 63.8623 - val_acc: 0.1911
    ## Epoch 17/30
    ## 0s - loss: 60.9271 - acc: 0.2325 - val_loss: 62.5651 - val_acc: 0.2799
    ## Epoch 18/30
    ## 0s - loss: 60.6013 - acc: 0.2104 - val_loss: 63.1358 - val_acc: 0.0989
    ## Epoch 19/30
    ## 0s - loss: 60.9249 - acc: 0.2285 - val_loss: 55.1772 - val_acc: 0.3461
    ## Epoch 20/30
    ## 0s - loss: 57.8489 - acc: 0.2685 - val_loss: 62.7711 - val_acc: 0.2220
    ## Epoch 21/30
    ## 0s - loss: 62.4322 - acc: 0.1543 - val_loss: 67.0223 - val_acc: 0.2364
    ## Epoch 22/30
    ## 0s - loss: 59.0988 - acc: 0.3066 - val_loss: 56.3118 - val_acc: 0.2188
    ## Epoch 23/30
    ## 0s - loss: 58.1392 - acc: 0.2425 - val_loss: 64.7004 - val_acc: 0.1797
    ## Epoch 24/30
    ## 0s - loss: 59.5388 - acc: 0.2144 - val_loss: 61.7567 - val_acc: 0.1746
    ## Epoch 25/30
    ## 0s - loss: 57.8003 - acc: 0.2605 - val_loss: 60.4283 - val_acc: 0.1173
    ## Epoch 26/30
    ## 0s - loss: 57.7487 - acc: 0.2285 - val_loss: 59.7629 - val_acc: 0.2097
    ## Epoch 27/30
    ## 0s - loss: 58.5914 - acc: 0.2305 - val_loss: 64.2916 - val_acc: 0.0944
    ## Epoch 28/30
    ## 0s - loss: 59.2083 - acc: 0.2285 - val_loss: 62.6276 - val_acc: 0.2251
    ## Epoch 29/30
    ## 0s - loss: 57.1372 - acc: 0.2625 - val_loss: 62.7807 - val_acc: 0.2426
    ## Epoch 30/30
    ## 0s - loss: 58.0849 - acc: 0.2405 - val_loss: 60.0844 - val_acc: 0.2935

``` r
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 28.260000"

#### Dropout

-   파라메터 수정을 하게 되면 어떠한 현상이 일어나는가?
-   테스트/학습시 dropout은 다르게 동작하는데, 특히 테스트의 경우 입력 노드의 숫자가 두배로 증가하기 때문에 입력 노드의 가중치에 0.5를 곱하는 후처리를 해줘야 된다.
    -   `keras`는 이러한 동작을 내부적으로 자동으로 처리해준다.

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 1024, activation = 'sigmoid') %>% 
  layer_dropout(rate=0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

# 모형을 생성한다. 
model <- keras_model(inputs = inputs, outputs = predictions)

#모형 구조 출력 
summary(model)
```

    ## Model
    ## ___________________________________________________________________________
    ## Layer (type)                     Output Shape                  Param #     
    ## ===========================================================================
    ## input_4 (InputLayer)             (None, 784)                   0           
    ## ___________________________________________________________________________
    ## dense_7 (Dense)                  (None, 1024)                  803840      
    ## ___________________________________________________________________________
    ## dropout_1 (Dropout)              (None, 1024)                  0           
    ## ___________________________________________________________________________
    ## dense_8 (Dense)                  (None, 10)                    10250       
    ## ===========================================================================
    ## Total params: 814,090
    ## Trainable params: 814,090
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
#컴파일 과정을 통해 최적화 조건을 선언한다. 
model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train_short, y_train_short,
    batch_size = 32,
    epochs = 30,
    verbose = 2,
    validation_split=0.9,
    callback_tensorboard(log_dir = "logs2/dropout")
  )
},type = 'stdout'))
```

    ## Train on 499 samples, validate on 4501 samples
    ## Epoch 1/30
    ## 0s - loss: 2.4647 - acc: 0.2285 - val_loss: 1.7450 - val_acc: 0.4395
    ## Epoch 2/30
    ## 0s - loss: 1.5327 - acc: 0.5110 - val_loss: 1.3272 - val_acc: 0.5939
    ## Epoch 3/30
    ## 0s - loss: 1.1860 - acc: 0.6353 - val_loss: 1.1401 - val_acc: 0.6701
    ## Epoch 4/30
    ## 0s - loss: 0.9534 - acc: 0.6934 - val_loss: 1.0631 - val_acc: 0.6652
    ## Epoch 5/30
    ## 0s - loss: 0.8049 - acc: 0.7495 - val_loss: 0.9144 - val_acc: 0.7394
    ## Epoch 6/30
    ## 0s - loss: 0.6815 - acc: 0.8116 - val_loss: 0.8490 - val_acc: 0.7656
    ## Epoch 7/30
    ## 0s - loss: 0.6401 - acc: 0.8056 - val_loss: 0.8113 - val_acc: 0.7645
    ## Epoch 8/30
    ## 0s - loss: 0.5556 - acc: 0.8417 - val_loss: 0.7690 - val_acc: 0.7832
    ## Epoch 9/30
    ## 0s - loss: 0.5033 - acc: 0.8617 - val_loss: 0.7553 - val_acc: 0.7794
    ## Epoch 10/30
    ## 0s - loss: 0.4982 - acc: 0.8617 - val_loss: 0.7349 - val_acc: 0.7843
    ## Epoch 11/30
    ## 0s - loss: 0.4426 - acc: 0.8958 - val_loss: 0.7047 - val_acc: 0.7947
    ## Epoch 12/30
    ## 0s - loss: 0.3985 - acc: 0.8958 - val_loss: 0.6851 - val_acc: 0.7969
    ## Epoch 13/30
    ## 0s - loss: 0.3768 - acc: 0.9098 - val_loss: 0.6890 - val_acc: 0.7887
    ## Epoch 14/30
    ## 0s - loss: 0.3130 - acc: 0.9279 - val_loss: 0.6692 - val_acc: 0.7998
    ## Epoch 15/30
    ## 0s - loss: 0.3028 - acc: 0.9259 - val_loss: 0.6670 - val_acc: 0.7918
    ## Epoch 16/30
    ## 0s - loss: 0.2975 - acc: 0.9399 - val_loss: 0.6532 - val_acc: 0.7943
    ## Epoch 17/30
    ## 0s - loss: 0.2881 - acc: 0.9359 - val_loss: 0.6457 - val_acc: 0.7987
    ## Epoch 18/30
    ## 0s - loss: 0.2592 - acc: 0.9279 - val_loss: 0.6302 - val_acc: 0.8029
    ## Epoch 19/30
    ## 0s - loss: 0.2405 - acc: 0.9479 - val_loss: 0.6175 - val_acc: 0.8076
    ## Epoch 20/30
    ## 0s - loss: 0.2276 - acc: 0.9559 - val_loss: 0.6078 - val_acc: 0.8105
    ## Epoch 21/30
    ## 0s - loss: 0.2123 - acc: 0.9579 - val_loss: 0.6077 - val_acc: 0.8096
    ## Epoch 22/30
    ## 0s - loss: 0.1979 - acc: 0.9619 - val_loss: 0.6081 - val_acc: 0.8123
    ## Epoch 23/30
    ## 0s - loss: 0.1852 - acc: 0.9639 - val_loss: 0.5956 - val_acc: 0.8127
    ## Epoch 24/30
    ## 0s - loss: 0.1930 - acc: 0.9599 - val_loss: 0.5860 - val_acc: 0.8145
    ## Epoch 25/30
    ## 0s - loss: 0.1829 - acc: 0.9579 - val_loss: 0.5909 - val_acc: 0.8096
    ## Epoch 26/30
    ## 0s - loss: 0.1932 - acc: 0.9599 - val_loss: 0.5855 - val_acc: 0.8152
    ## Epoch 27/30
    ## 0s - loss: 0.1449 - acc: 0.9780 - val_loss: 0.5822 - val_acc: 0.8163
    ## Epoch 28/30
    ## 0s - loss: 0.1571 - acc: 0.9739 - val_loss: 0.5803 - val_acc: 0.8129
    ## Epoch 29/30
    ## 0s - loss: 0.1560 - acc: 0.9639 - val_loss: 0.5700 - val_acc: 0.8194
    ## Epoch 30/30
    ## 0s - loss: 0.1332 - acc: 0.9800 - val_loss: 0.5680 - val_acc: 0.8238

``` r
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 81.730000"

#### Combining different types of regularization

-   L2 and Dropout

``` r
inputs <- layer_input(shape = c(784))
 
predictions <- inputs %>%
  layer_dense(units = 2048, activation = 'sigmoid',kernel_regularizer=regularizer_l2(0.1)) %>% 
  layer_dropout(rate=0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

# 모형을 생성한다. 
model <- keras_model(inputs = inputs, outputs = predictions)

#모형 구조 출력 
summary(model)
```

    ## Model
    ## ___________________________________________________________________________
    ## Layer (type)                     Output Shape                  Param #     
    ## ===========================================================================
    ## input_5 (InputLayer)             (None, 784)                   0           
    ## ___________________________________________________________________________
    ## dense_9 (Dense)                  (None, 2048)                  1607680     
    ## ___________________________________________________________________________
    ## dropout_2 (Dropout)              (None, 2048)                  0           
    ## ___________________________________________________________________________
    ## dense_10 (Dense)                 (None, 10)                    20490       
    ## ===========================================================================
    ## Total params: 1,628,170
    ## Trainable params: 1,628,170
    ## Non-trainable params: 0
    ## ___________________________________________________________________________
    ## 
    ## 

``` r
sgd_lr <- optimizer_sgd(lr=0.01)
#컴파일 과정을 통해 최적화 조건을 선언한다. 
model %>% compile(
  optimizer = sgd_lr,
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#학습 
cat(py_capture_output({
  history <- model %>% fit(
    x_train_short, y_train_short,
    batch_size = 32,
    epochs = 30,
    verbose = 2,
    validation_split=0.9,
    callback_tensorboard(log_dir = "logs2/l2_dropout")
  )
},type = 'stdout'))
```

    ## Train on 499 samples, validate on 4501 samples
    ## Epoch 1/30
    ## 0s - loss: 112.1988 - acc: 0.3226 - val_loss: 107.7501 - val_acc: 0.5312
    ## Epoch 2/30
    ## 0s - loss: 104.3865 - acc: 0.6774 - val_loss: 100.7643 - val_acc: 0.7178
    ## Epoch 3/30
    ## 0s - loss: 97.6950 - acc: 0.7856 - val_loss: 94.4947 - val_acc: 0.7343
    ## Epoch 4/30
    ## 0s - loss: 91.5106 - acc: 0.8637 - val_loss: 88.6405 - val_acc: 0.7425
    ## Epoch 5/30
    ## 0s - loss: 85.7811 - acc: 0.8717 - val_loss: 83.1100 - val_acc: 0.7863
    ## Epoch 6/30
    ## 0s - loss: 80.4516 - acc: 0.8818 - val_loss: 77.9692 - val_acc: 0.7978
    ## Epoch 7/30
    ## 0s - loss: 75.4161 - acc: 0.9238 - val_loss: 73.1607 - val_acc: 0.8020
    ## Epoch 8/30
    ## 0s - loss: 70.7030 - acc: 0.9459 - val_loss: 68.6723 - val_acc: 0.7932
    ## Epoch 9/30
    ## 0s - loss: 66.3231 - acc: 0.9459 - val_loss: 64.4085 - val_acc: 0.8183
    ## Epoch 10/30
    ## 0s - loss: 62.1960 - acc: 0.9519 - val_loss: 60.4657 - val_acc: 0.8092
    ## Epoch 11/30
    ## 0s - loss: 58.3136 - acc: 0.9739 - val_loss: 56.7211 - val_acc: 0.8263
    ## Epoch 12/30
    ## 0s - loss: 54.6940 - acc: 0.9800 - val_loss: 53.2291 - val_acc: 0.8263
    ## Epoch 13/30
    ## 0s - loss: 51.2856 - acc: 0.9880 - val_loss: 49.9624 - val_acc: 0.8274
    ## Epoch 14/30
    ## 0s - loss: 48.1141 - acc: 0.9860 - val_loss: 46.8887 - val_acc: 0.8271
    ## Epoch 15/30
    ## 0s - loss: 45.1247 - acc: 0.9840 - val_loss: 44.0231 - val_acc: 0.8236
    ## Epoch 16/30
    ## 0s - loss: 42.3251 - acc: 0.9900 - val_loss: 41.3173 - val_acc: 0.8285
    ## Epoch 17/30
    ## 0s - loss: 39.7111 - acc: 0.9880 - val_loss: 38.7896 - val_acc: 0.8265
    ## Epoch 18/30
    ## 0s - loss: 37.2420 - acc: 0.9840 - val_loss: 36.4155 - val_acc: 0.8278
    ## Epoch 19/30
    ## 0s - loss: 34.9385 - acc: 0.9840 - val_loss: 34.1767 - val_acc: 0.8365
    ## Epoch 20/30
    ## 0s - loss: 32.7591 - acc: 0.9940 - val_loss: 32.0929 - val_acc: 0.8327
    ## Epoch 21/30
    ## 0s - loss: 30.7229 - acc: 0.9980 - val_loss: 30.1240 - val_acc: 0.8374
    ## Epoch 22/30
    ## 0s - loss: 28.8185 - acc: 0.9960 - val_loss: 28.2957 - val_acc: 0.8323
    ## Epoch 23/30
    ## 0s - loss: 27.0290 - acc: 0.9920 - val_loss: 26.5679 - val_acc: 0.8360
    ## Epoch 24/30
    ## 0s - loss: 25.3539 - acc: 0.9960 - val_loss: 24.9383 - val_acc: 0.8389
    ## Epoch 25/30
    ## 0s - loss: 23.7868 - acc: 0.9980 - val_loss: 23.4229 - val_acc: 0.8411
    ## Epoch 26/30
    ## 0s - loss: 22.3167 - acc: 0.9940 - val_loss: 22.0000 - val_acc: 0.8418
    ## Epoch 27/30
    ## 0s - loss: 20.9232 - acc: 1.0000 - val_loss: 20.6655 - val_acc: 0.8385
    ## Epoch 28/30
    ## 0s - loss: 19.6278 - acc: 1.0000 - val_loss: 19.4157 - val_acc: 0.8396
    ## Epoch 29/30
    ## 0s - loss: 18.4166 - acc: 0.9960 - val_loss: 18.2294 - val_acc: 0.8451
    ## Epoch 30/30
    ## 0s - loss: 17.2699 - acc: 1.0000 - val_loss: 17.1393 - val_acc: 0.8385

``` r
accuracy(x_test, y_test, model)
```

    ## [1] "Accuracy on data is: 83.850000"

``` r
#텐서보드 띄워 보기 
tensorboard(log_dir = 'lectures/logs2/',host = '0.0.0.0', port = 8002)
```
