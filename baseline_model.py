import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector

from tensorflow.keras.models import Sequential
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.utils.vis_utils import plot_model
from keras.models import load_model


def transfer_to_X_Y(data, df_features, data_paras):
    
    know = data_paras['know']
    know_det = data_paras['know_det']
    unknow = data_paras['unknow']
    unknow_det = data_paras['unknow_det']
    step = data_paras['step']
    df_features = np.array(df_features)
    
    X = []
    Y = []
    IDX = []
    for idx in range(0, data.shape[0],step):
        if idx + know + unknow + unknow_det> data.shape[0]:
            break
        x = data[idx:idx+know+know_det:know_det, :]
        if df_features != []:
            xf = df_features[idx:idx+know+know_det:know_det, :]
            x = np.c_[x, xf]
        y = data[idx+know:idx+know+unknow+unknow_det:unknow_det, 1]

        if idx % 100000 ==0:
            print(idx)
        X.append(x)
        Y.append(y)
        IDX.append(idx)
    print('sample:',len(IDX))
    # reshape Y from 2D to 3D
    Y = reshape_2D_to_3D(np.array(Y))
    return (np.array(X),Y,IDX)

def df_to_array(df):
    N = 1440
    df.columns = ['Timestamp','Water_Consumption']
    idx = np.arange(1,N+1).tolist()
    num_day = df.shape[0] // N
    idx = np.array(idx * num_day)
    w = np.array(df['Water_Consumption'])
    data = np.c_[idx, w]
    return data

def build_model_timedistribute(xy, model_paras):
    
    epochs = model_paras['epochs']
    batch_size = model_paras['batch_size']
    model_name = model_paras['model_name']
    
    train_x, train_y, dev_x, dev_y = xy
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(TimeDistributed(Dense(train_y.shape[2])))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse', 'mape'])
    print(model.summary())

#     plot_model(model, to_file=f'{model_name}.png', show_shapes=True, 
#                show_layer_names=True, expand_nested=True,show_dtype=True)

# 	# fit network
#     history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, 
#                         validation_data=(dev_x, dev_y), verbose=2, shuffle=False)
# 	# plot history
#     # fig = plt.Figure()
#     plt.plot(history.history['loss'], label='Train')
#     plt.plot(history.history['val_loss'], label='Validation')
#     plt.legend()
#     model.save(f'{model_name}')


# normalize the X, Y
def normalize(X, Y):
    max_X = np.max(np.max(X, axis=0), axis=0)
    min_X = np.min(np.min(X, axis=0), axis=0)
    max_Y = max_X[1]
    min_Y = min_X[1]
    
    X = (X - min_X) / (max_X - min_X)
    Y = (Y - min_Y) / (max_Y - min_Y)
    
    norm_paras = {'max_X':max_X, 'min_X':min_X,
                 'max_Y':max_Y, 'min_Y':min_Y}
    
    return X, Y, norm_paras

# transfer x, y to the raw scale
def inverse_normalize(x, y, norm_paras):
    max_X = norm_paras['max_X']
    min_X = norm_paras['min_X']
    max_Y = norm_paras['max_Y']
    min_Y = norm_paras['min_Y']
    
    X = x * (max_X - min_X) + min_X
    Y = y * (max_Y - min_Y) + min_Y
    return X, Y

# reshape y from 2D to 3D
def reshape_2D_to_3D(y):
    return np.array([yi.reshape((len(yi),-1)) for yi in y])

# reshape y from 3D to 2D
def reshape_3D_to_2D(y):
    return y.reshape((y.shape[0], y.shape[1]))
    
    
def split(X, Y, split_ratio):
    split_ratio = np.array(split_ratio)
    NUM_SAMPLE = X.shape[0]
    idx = [int(x) for x in np.ceil(np.cumsum(NUM_SAMPLE*split_ratio))]
    
    train = (X[:idx[0]], Y[:idx[0]], IDX[:idx[0]])
    dev = (X[idx[0]:idx[1]], Y[idx[0]:idx[1]], IDX[idx[0]:idx[1]])
    test = (X[idx[1]:], Y[idx[1]:], IDX[idx[1]:])
    
    return {'train': train, 'dev': dev, 'test': test}


def evaluate(model, x, y, norm_paras):
    yhat = model.predict(x)
    if yhat.shape[2] != 1:
        yhat = np.mean(yhat, axis=2)
    
    _, y = inverse_normalize(x, y, norm_paras)
    _, yhat = inverse_normalize(x, yhat, norm_paras)
    
    # rmse = sqrt(mean_squared_error(reshape_3D_to_2D(y), reshape_3D_to_2D(yhat)))
    y_2d = reshape_3D_to_2D(y)
    yhat_2d = reshape_3D_to_2D(yhat)
    
    rmse = mean_squared_error(y_2d, yhat_2d,squared=False)
    mae = mean_absolute_error(y_2d, yhat_2d)
    mape = np.mean(np.abs(y_2d-yhat_2d) / y_2d)
    
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('MAPE:', mape)
    return yhat,(rmse, mae, mape)

def plot_x_y_test(X, Y, Yhat, Idx, data_paras, 
                  norm_paras, model_name,plot_test_i=0):
    
    X, Y = inverse_normalize(X, Y, norm_paras)
    know = data_paras['know']
    know_det = data_paras['know_det']
    unknow = data_paras['unknow']
    unknow_det = data_paras['unknow_det']
    
    plt.close("all")
    plt.figure(figsize = (15,5))
    
    i=plot_test_i
    x, y, yhat, idx = X[i], Y[i], Yhat[i], Idx[i]
    
    know_t = np.arange(0, know+know_det, know_det)

    plt.plot(know_t,x[:,1], 'g*-')

    unknow_t = np.arange(know, know+unknow+unknow_det, unknow_det)
    plt.plot(unknow_t, y, 'go-')

#
    plt.plot(data[idx:idx+know+unknow+unknow_det,1],'k',alpha = .3)
    plt.plot(unknow_t,yhat, 'r*-')


    xticks = np.arange(0,know+unknow, 180)
    t = [x[11:] for x in df['Timestamp'][idx:idx+know+unknow:180]]

    plt.xticks(ticks=xticks, labels=t)
    plt.grid('minor')
    plt.xlabel('Time (min)')
    plt.ylabel('Water Consumption')
    plt.savefig(f'test plot/{model_name}_{i}.png')

def plot_multiple_test(X, Y, Idx, data_paras, norm_paras, plot_test_i=0):
    
    know = data_paras['know']
    know_det = data_paras['know_det']
    unknow = data_paras['unknow']
    unknow_det = data_paras['unknow_det']
    
    i = plot_test_i

    yhats, errs = [], []
    
    model_names = ['Repeat', 'Dense', 'CNN', 'LSTM','GRU', 'Bi-LSTM']
    # repeat model
    x_temp, _ = inverse_normalize(X, Y, norm_paras)
    yhats.append(x_temp[i,:,1])
    
    # Dense model
    model = tf.keras.models.load_model('model/Dense_model')
    Yhat, _ = evaluate(model, X, Y, norm_paras)
    yhats.append(Yhat[i])
    
    # CNN model
    model = tf.keras.models.load_model('model/CNN_model')
    Yhat, _ = evaluate(model, X, Y, norm_paras)
    yhats.append(Yhat[i])
    
    # LSTM model
    model = tf.keras.models.load_model('model/LSTM_model')
    Yhat, _ = evaluate(model, X, Y, norm_paras)
    yhats.append(Yhat[i])
    
    # GRU model
    model = tf.keras.models.load_model('model/GRU_model')
    Yhat, _ = evaluate(model, X, Y, norm_paras)
    yhats.append(Yhat[i])
    
    # BiLstm model
    model = tf.keras.models.load_model('model/BiLstm_model')
    Yhat, _ = evaluate(model, X, Y, norm_paras)
    yhats.append(Yhat[i])
    
    
    X, Y = inverse_normalize(X, Y, norm_paras)
    x, y, idx = X[i], Y[i], Idx[i]
    
    
    plt.close("all")
    # plt.figure(figsize = (15,5))
    plt.rcParams["figure.figsize"] = (15,6)
    plt.rcParams['font.size'] = '20'
    
    
    know_t = np.arange(0, know+know_det, know_det)
    unknow_t = np.arange(know, know+unknow+unknow_det, unknow_det)
    
    plt.plot(know_t,x[:,1], 'g*-',alpha = .3)
    # plt.plot(unknow_t, y, 'go-')

    plt.plot(data[idx:idx+know+unknow+unknow_det,1],'k',alpha = .2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for j, yhat in enumerate(yhats):
        plt.plot(unknow_t,yhat, color = colors[j], label = model_names[j], linewidth = 2, alpha=1)


    xticks = np.arange(0,know+unknow, 360)
    t = [x[11:] for x in df['Timestamp'][idx:idx+know+unknow:360]]

    plt.xticks(ticks=xticks, labels=t)
    plt.legend(loc='upper left')
    plt.grid('minor')
    plt.xlabel('Time (min)')
    plt.ylabel('Water Consumption')
    plt.savefig(f'test_{i}.png', dpi=900)
    
# create some directories 
def create_dir():
    # save model
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('test plot'):
        os.mkdir('test plot')
    if not os.path.exists('model plot'):
        os.mkdir('model plot')
    if not os.path.exists('model loss'):
        os.mkdir('model loss')
    
def fit_model(model, data, model_paras):
    
    train_x, train_y, dev_x, dev_y = data
    model_name = model_paras['model_name']
    epochs = model_paras['epochs']
    batch_size = model_paras['batch_size']
    model_name = model_paras['model_name']
    
    	# fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, 
                        validation_data=(dev_x, dev_y), verbose=2, shuffle=False)
    print(model.summary())

	# plot history
    # fig = plt.Figure()
    plt.close('all')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.savefig(f'model loss/{model_name}.png')
    model.save(f'model/{model_name}')
    tf.keras.utils.plot_model(model,to_file = f'model plot/{model_name}.png',show_shapes=True,)
    return model


class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
      outputs = inputs[:,:,1]
      outputs = tf.expand_dims(outputs, -1)
      return outputs

def Repeat_model(data, error, norm_paras, data_paras, model_paras):
    
    model_name = model_paras['model_name']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    # repeat baseline model
    model = RepeatBaseline()
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    
    tf.keras.utils.plot_model(model, to_file=f'model plot/{model_name}.png',
                              show_shapes=True)
    # fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    # model.save(f'model/{model_name}')
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def Linear_model(data, error, norm_paras, data_paras, model_paras):
    
    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, time*features]
        tf.keras.layers.Flatten(),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(out_steps*out_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, out_features])
    ])
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    
    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def Dense_model(data, error, norm_paras, data_paras, model_paras):

    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, time*features]
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,
                              activation='relu'),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(out_steps*out_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, out_features])
    ])
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def CNN_model(data, error, norm_paras, data_paras, model_paras):
   
    width = 5
    in_steps = model_paras['in_steps']
    in_features = model_paras['in_features']
    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -width:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(width)),
        # tf.keras.layers.Dense(512,
        #                       activation='relu'),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(out_steps*out_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, out_features])
    ])
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def LSTM_model(data, error, norm_paras, data_paras, model_paras):
    
    in_steps = model_paras['in_steps']
    in_features = model_paras['in_features']
    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    
    model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.LSTM(128, return_sequences=False),
    # tf.keras.layers.Dense(512,
    #                       activation='relu'),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(out_steps*out_features,
                          # activation='sigmoid',
                          # kernel_initializer=tf.initializers.zeros(),
                          ),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([out_steps, out_features])
    ])
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    
    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def GRU_model(data, error, norm_paras, data_paras, model_paras):
    
    in_steps = model_paras['in_steps']
    in_features = model_paras['in_features']
    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    
    model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.GRU(128, return_sequences=False),
    # tf.keras.layers.Dense(512,
    #                       activation='relu'),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(out_steps*out_features,
                          # activation='sigmoid',
                          # kernel_initializer=tf.initializers.zeros(),
                          ),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([out_steps, out_features])
    ])
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    
    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def BiLstm_model(data, error, norm_paras, data_paras, model_paras):
    
    in_steps = model_paras['in_steps']
    in_features = model_paras['in_features']
    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    
    model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Bidirectional(LSTM(128)),
    # tf.keras.layers.Dense(512,
    #                       activation='relu'),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(out_steps*out_features,
                          # activation='sigmoid',
                          # kernel_initializer=tf.initializers.zeros(),
                          ),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([out_steps, out_features])
    ])
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])
    
    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error

def TimeDistributed_LSTM_model(data, error, norm_paras, data_paras, model_paras):
    
    in_steps = model_paras['in_steps']
    in_features = model_paras['in_features']
    out_steps = model_paras['out_steps']
    out_features = model_paras['out_features']
    loss = model_paras['loss']
    optimizer = model_paras['optimizer']
    model_name = model_paras['model_name']
    print(f'{model_name}')
    
    train_x, train_y, train_idx = data[0]
    dev_x, dev_y, dev_idx = data[1]
    test_x, test_y, test_idx = data[2]
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(in_steps, in_features), return_sequences=True))
    model.add(TimeDistributed(Dense(out_features)))
    # model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['mse'])
    model.compile(loss=loss, optimizer=optimizer,metrics=['mae','mape'])

    model = fit_model(model,(train_x, train_y, dev_x, dev_y), model_paras)
    
    #evaluate model
    _,train_error = evaluate(model, train_x, train_y, norm_paras)
    _,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
    yhat,test_error = evaluate(model, test_x, test_y, norm_paras)
    
    error[f'{model_name}'] = [train_error, dev_error, test_error]
    
    print(model.summary())
    
    plot_x_y_test(test_x, test_y, yhat, test_idx, 
                  data_paras,norm_paras, model_name, plot_test_i=0)
    
    return error


# def plot_error(error, metrics='RMSE'):
#     x = np.arange(len(error.columns))
    
#     width = 0.3
#     # if metrics == 'RMSE':
#     #     i = 0
#     # if metrics == 'MAE':
#     #     i = 1
#     # if metrics == 'MAPE':
#     #     i = 2
#     # dev = [d[i] for d in error.loc['dev',]]
#     # test = [t[i] for t in error.loc['test',]]
    
#     dev = error.loc['dev',]
#     test = error.loc['test',]
    
#     plt.rcParams["figure.figsize"] = (8,6)
#     plt.rcParams['font.size'] = '16'

#     plt.bar(x - 0.17, dev, width, label='Dev')
#     plt.bar(x + 0.17, test, width, label='Test')
#     print(test)
#     plt.xticks(ticks=x, labels=[e[:-6] for e in error.columns],
#            rotation=45)
#     plt.ylabel(f'{metrics} (average over all times and outputs)')
#     _ = plt.legend()
#     plt.grid('minor')
#     plt.savefig(f'error_{metrics}.png',dpi=900)
    
    
#  config GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
        
data_paras = dict(step = 5,
                  know = 1440,
                  unknow = 1440,
                  unknow_det = 20,
                  know_det = 20)



    

# split samples into train, dev, test
split_ratio = [0.8, 0.1, 0.1]

# load data
df=pd.read_csv('clean data/HY.csv',header=0)
data = df_to_array(df)

# covariate features
covariate_features = ['MaxTemp','MinTemp','IW', 'HF_exponential_kernel']
df_features = pd.read_csv('new_Features.csv',header=0)
df_features = df_features[covariate_features]

# extract samples: X,Y
(X,Y,IDX) = transfer_to_X_Y(data, df_features, data_paras)

# normalize Y
X, Y, norm_paras = normalize(X, Y)

# split train, dev, test set
train_x, train_y, train_idx = split(X, Y, split_ratio)['train']
dev_x, dev_y, dev_idx = split(X, Y, split_ratio)['dev']
test_x, test_y, test_idx = split(X, Y, split_ratio)['test']

train = train_x, train_y, train_idx
dev = dev_x, dev_y, dev_idx
test = test_x, test_y, test_idx

# create some directories
create_dir()

# inputs and outputs dim
_, in_steps, in_features = train_x.shape
_, out_steps, out_features = train_y.shape

model_paras = dict(epochs = 100,
                   batch_size = 256,
                   model_name = ' ',
                   in_steps = in_steps,
                   in_features = in_features,
                   out_steps = out_steps,
                   out_features = out_features,
                   loss = 'mean_squared_error',
                   optimizer = 'adam')

# performance will save the accuracy of each model
error = pd.DataFrame(None, index=['train', 'dev', 'test'])

model_paras['model_name'] = 'Repeat_model'
Repeat_model((train, dev, test), error, norm_paras, data_paras, model_paras)

# model_paras['model_name'] = 'Linear_model'
# Linear_model((train, dev, test), error, norm_paras, data_paras, model_paras)

model_paras['model_name'] = 'Dense_model'
Dense_model((train, dev, test), error, norm_paras, data_paras, model_paras)

model_paras['model_name'] = 'CNN_model'
CNN_model((train, dev, test), error, norm_paras, data_paras, model_paras)

model_paras['model_name'] = 'LSTM_model'
LSTM_model((train, dev, test), error, norm_paras, data_paras, model_paras)


model_paras['model_name'] = 'GRU_model'
GRU_model((train, dev, test), error, norm_paras, data_paras, model_paras)

# model_paras['model_name'] = 'TimeDistributed_LSTM_model'
# TimeDistributed_LSTM_model((train, dev, test), error, norm_paras, data_paras, model_paras)

model_paras['model_name'] = 'BiLstm_model'
BiLstm_model((train, dev, test), error, norm_paras, data_paras, model_paras)


model_name = 'BiLstm_model'
#evaluate model
model = load_model('model/BiLstm_model - Copy')
_,train_error = evaluate(model, train_x, train_y, norm_paras)
_,dev_error = evaluate(model, dev_x, dev_y, norm_paras)
yhat,test_error = evaluate(model, test_x, test_y, norm_paras)

error[f'{model_name}'] = [train_error, dev_error, test_error]

print(model.summary())

plot_x_y_test(test_x, test_y, yhat, test_idx, 
              data_paras,norm_paras, model_name, plot_test_i=0)

print(error)

error.to_csv('error.csv')

error1 = pd.read_excel('error_RMSE.xlsx', index_col=0)


# plot_error(error1[['Repeat_model', 'Dense_model', 'CNN_model', 'LSTM_model', 'GRU_model','BiLstm_model']])


