from math import sqrt

from keras.layers import Input, Dense, LSTM, merge, Conv1D, Dropout, Bidirectional, Multiply, TimeDistributed
from keras.losses import mean_squared_error
from keras.models import Model

import matplotlib,os
import tensorflow as tf
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error

matplotlib.use("TkAgg")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为1，2号的GPU
config = tf.ConfigProto(allow_soft_placement=True)  #动态分配
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 每个GPU现存上届控制在60%以内
# session = tf.Session(config=config)
import keras.backend.tensorflow_backend as KTF
# 设置session
KTF.set_session(session)
from attention_file import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import  pandas as pd
import  numpy as np





SINGLE_ATTENTION_VECTOR = False
# def attention_3d_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = inputs
#     #a = Permute((2, 1))(inputs)
#     #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(input_dim, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((1, 2), name='attention_vec')(a)
#
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul
def attention_3d_block(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul



def create_dataset(dataset, look_back):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

#多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    x = Dropout(0.3)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_out=LSTM(lstm_units,return_sequences=True)(x)
    #lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def model_attention_applied_before_lstm():
    # K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 50
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_attention_applied_after_lstm():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS,))
    lstm_units = 50
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_xiugai():
    lstm_units = 50
    model = Sequential()
    model.add(LSTM(lstm_units, dropout=0.64,
               input_shape=(3, 4)))
    model.add(RepeatVector(3))
    model.add(LSTM(lstm_units, dropout=0.64, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='tanh')))
    return model
#加载数据

# data = pd.read_csv("g:/features_extraction(selected_images).csv")
data = pd.read_csv("./zhu/20190101.csv")
# data = data.drop(['date','wnd_dir'], axis = 1)

print(data.columns)
print(data.shape)


INPUT_DIMS = 4
TIME_STEPS = 3
lstm_units = 64

#归一化
data,normalize = NormalizeMult(data)
pollution_data = data[:,0].reshape(len(data),1)

train_X, _ = create_dataset(data,TIME_STEPS)
_ , train_Y = create_dataset(pollution_data,TIME_STEPS)
# test_X_MA=train_X[:100,:]  #测试集
# test_Y_MA=train_Y[:100,:]

print(train_X.shape,train_Y.shape)

# m = model_attention_applied_after_lstm()
m=model_xiugai()
m.summary()
m.compile(optimizer='adam', loss='mse')
epochs=3
batchsize=32
history=m.fit(train_X[:8000,:], train_Y[:8000,:], epochs=epochs, batch_size=batchsize, validation_split=0.2)
m.save("./lstm+attention_epoch_{}-batchsize_{}_step_{}_16000_model.h5".format(epochs,batchsize,TIME_STEPS))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
model = load_model("./lstm+attention_epoch_{}-batchsize_{}_step_{}_16000_model.h5".format(epochs,batchsize,TIME_STEPS))
trainPredict = model.predict(train_X[15457:,:])   ##将所有数据分为训练集和验证集，测试集
trainPredict = FNormalizeMult(trainPredict,normalize)
trainY = FNormalizeMult(train_Y[15457:,:],normalize)
pyplot.plot(trainY,label='Actual')
pyplot.plot(trainPredict,label='Predict')
pyplot.xlabel('Time-Number')
pyplot.ylabel('CNN_Lstm_attention Photovaltaic Power [W/m^2]')
pyplot.legend(loc='upper left')
pyplot.show()

train_Y=trainY.reshape(-1)
trainPredict=trainPredict.reshape(-1)

print("train_Y.shape",train_Y.shape)
print("trainPredict.shape",trainPredict.shape)
# print(train_Y[:,0])
# train_Y=np.array(train_Y)
# trainPredict=np.array(trainPredict)
print(train_Y)
def write_to_txt(inv_yhat,inv_y):
    with open('lstm_attention__predict_{}_{}_{}.txt'.format(epochs,batchsize,TIME_STEPS),'w') as f:
        for x in inv_yhat:
            f.write(str(x))
            f.write('\n')
    with open('lstm_attention__actual_{}_{}_{}.txt'.format(epochs,batchsize,TIME_STEPS),'w') as f:
        for x in inv_y:
            f.write(str(x))
            f.write('\n')
write_to_txt(trainPredict,train_Y)
def rMSE(train_Y,trainPredict):
    sum = 0
    for i in range(len(train_Y)):
        a=(train_Y[i]-trainPredict[i])*(train_Y[i]-trainPredict[i])
        sum=sum+a
    print("sum",sum)
    rmse=sqrt(sum/len(train_Y))
    return rmse
rmse = rMSE(trainY, trainPredict)
# print('Test RMSE: %.3f' % rmse)
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)

mape=MAPE(train_Y,trainPredict)
print("mean_absolute_error:", mean_absolute_error(train_Y, trainPredict))
R_Squared=1- mean_squared_error(train_Y,trainPredict)/ np.var(train_Y)

print('rmse: %.3f' % rmse)
print('mape: %.3f' % mape)
print("r_squared: %.3f" % R_Squared)
#np.save("normalize.npy",normalize)