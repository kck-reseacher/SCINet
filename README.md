# SCINet
Sample Convolution and Interaction Networks


# Requirements
tensorflow=2.6.2
pandas=1.1.5
numpy=1.19.5
scikit-learn=0.24.2
giotto-tda=0.5.1
plotly=5.10.0
importlib=1.0.4

# How to Training

$ python train.py -d {train data directory} -g {gpu number} -m {module name}
# example
$ python train.py -d /home/kck/home/train_data -g 0 -m analyzer

# How to Predict

$ python predict.py -d {train data directory} -m {module name}
# example
$ python predict.py -d /home/kck/home/train_data -m analyzer

# predict result
## BTC-USD Close feature

![](C:\Users\cksty\Downloads\BTC-USD_Close.png)