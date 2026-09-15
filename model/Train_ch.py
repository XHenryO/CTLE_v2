"""
Created on '2021/10/22'

Used for Channel Estimation

@author: JY
Version: v1.9
History:
2022/06/21 v1.1: Re-load and rename;
2022/07/20 v1.2: add BatchNormalization
2022/09/23 v1.3: need random.seed(SEED) to control
2022/10/12 v1.4: add attention
2022/10/13 v1.5: add time slot input to generate 13GHz
2022/10/14 v1.6: add activation function "noisy linear"
2022/10/24 v1.7: plot loss use log10
2022/10/25 v1.8: add ModelCheckpoint to get weights of each epoch; add K.function to get output of any nodes
2023/01/05 v1.9: add keras_flops to calculate the flops, but not support RNN-like NN; For RNN-like https://github.com/mlperf-deepts/nns

"""
# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Multiply
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
import numpy as np
import random
from SelfDesign import ANN_data_transfer, MyLayer

# from tfdeterminism import patch
# from keras_flops import get_flops

# Make sure each time is the same
# Setting seed value
# from https://stackoverflow.com/a/52897216
# generated randomly by running `random.randint(0, 100)` once
SEED = 10
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)
# patch()  # force modify this to support 2.5: if re.match("(1\.(14|15)|2\.(0|5))", tf_version):

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1 for CPU;


retrain = 1  # 1 for Post-equ; 0 for Pre-equ
train_gan=1
channelestForPost = 0  # 1 for Equalization; 0 for channel estimation
# Generate dummy data
for i in range(1):
    if channelestForPost:
        if retrain:
            ExtraFlag = 'Rxdata'     ## 淇濆瓨鏂囦欢鐨勫悕绉?
        else:
            ExtraFlag = 'Txdata'
    else:
        ExtraFlag = 'Txdata'

    Extra = 'channelest_TTHNet_Sin'
    epoch =30
    batch_size = 256
    ratio = 0.8  # ratio of
    upsam=4
    # output_dim = 21*upsam #3 5
    output_dim = 85
    # node = 2**(i+1)
    # taps =39*upsam  #197 165 61
    taps = 157
    # name of file
    path = './channel/'
    # name_Tx = ['./Txdata_NN_baseline', '.txt']
    name_Tx = ['D:/CTLE_v2/data/Txdata_NN_baseline_wgn', '.txt']     ## 瀵煎叆鏂囦欢鐨勫悕绉?
    # name_Tx = ['./MLP_tx', Extra, '.txt']
    name_Tx = ''.join(name_Tx)
    # name_Rx = ['./MLP_rx', Extra, '.txt']
    # name_Rx = ['./Rxdata_NN_baseline', '.txt']
    name_Rx = ['D:/CTLE_v2/data/Rxdata_NN_baseline_wgn', '.txt']
    name_Rx = ''.join(name_Rx)
    # name_Sa = ['Rxdata_after', str(Extra), '.txt']
    name_Sa = [ExtraFlag, '_afterNN', '.txt']
    name_Sa = ''.join(name_Sa)
    # data pre-process
    # origindata_path = '''../../../Downloads\\'''
    # data_Tx = np.loadtxt(origindata_path + name_Tx)
    # data_Rx = np.loadtxt(origindata_path + name_Rx)

    data_Tx = np.loadtxt(name_Tx)
    data_Rx = np.loadtxt(name_Rx)
    ## 涓存椂鏇挎崲            鍏朵綑鏀瑰姩锛歮atlabA4鐨勭敓鎴愪唬鐮侊紝鐢婚璋变互鍙婃敼淇濆瓨鏂囦欢  绾挎€у浘         ##  杩欓噷瀛AN鏁堟灉寰堝ソ
    # data_Tx=np.loadtxt('G:\WY_results\Channel--GAN/FSOdata/FSO/channel_gan_fso/Tx/simu_A0_in_32QAM_wu0.txt')
    # data_Rx=np.loadtxt('G:\WY_results\Channel--GAN/FSOdata/FSO/channel_gan_fso/Rx/expe_32QAM_170.txt')

    data_Tx = data_Tx / np.max(abs(data_Tx))
    data_Rx = data_Rx / np.max(abs(data_Rx))
    # x_Rx_2 = data_Rx[:, np.newaxis]  #discard
    # x_Rx_2 = data_Rx.reshape(-1, 1)   #discard
    # frequency response
    # TX
    # data_Tx_fft = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data_Tx))))
    # plt.figure()
    # plt.plot(data_Tx_fft, '-b')
    # plt.title('Tx')
    # # Rx
    # data_Rx_fft = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data_Rx))))
    # plt.figure()
    # plt.plot(data_Rx_fft, '-r')
    # plt.title('Rx')

    # prepare the data for ANN
    data_Tx_length = len(data_Tx)
    data_Rx_length = len(data_Rx)

    if channelestForPost:
        x_train_temp = data_Rx[0:int(data_Rx_length * ratio)]  # 杈撳叆涓篟x
        y_train_temp = data_Tx[0:int(data_Tx_length * ratio)]  # 鏍囩涓篢x
        x_test_temp = data_Rx[int(data_Rx_length * ratio) + 1:]
        y_test_temp = data_Tx[int(data_Tx_length * ratio) + 1:]
    else:
        x_train_temp = data_Tx[0:int(data_Tx_length * ratio)]  # 杈撳叆涓篢x
        y_train_temp = data_Rx[0:int(data_Rx_length * ratio)]  # 鏍囩涓篟x
        x_test_temp = data_Tx[int(data_Tx_length * ratio) + 1:]
        y_test_temp = data_Rx[int(data_Rx_length * ratio) + 1:]

    # train sequence and label
    x_train = ANN_data_transfer(x_train_temp, taps)
    if taps == output_dim:
        y_train_temp2 = y_train_temp
    else:   ## 杩欓噷涓嶈ˉ鐩存帴鍘婚櫎锛岄娴嬫椂鍐嶈ˉ
        y_train_temp2 = y_train_temp[int((taps - output_dim) / 2):int(-(taps - output_dim) / 2)]
    y_train = ANN_data_transfer(y_train_temp2, output_dim)

    x_test = ANN_data_transfer(x_test_temp, taps)
    if taps == output_dim:
        y_test_temp2 = y_test_temp
    else:
        y_test_temp2 = y_test_temp[int((taps - output_dim) / 2):int(-(taps - output_dim) / 2)]
    y_test = ANN_data_transfer(y_test_temp2, output_dim)
    # test sequence and label
    # 褰撴ā鍨嬭缁冭繃绋嬩腑锛岄獙璇侀泦涓婄殑val_mse杩炵画涓€娈垫椂闂存病鏈夋槑鏄惧彉濂斤紝灏辨妸瀛︿範鐜囪皟灏忎竴鐐广€?
    # factor=0.9, 姣忔闄嶄綆鏃跺彉鎴?.9鍊嶏紝verbose=1:瑙﹀彂闄嶄綆鏃舵墦鍗帮紱patience=5锛氳繛缁?娆al_mse娌℃湁鏀瑰杽鎵嶆墽琛岄檷浣?
    reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.9, verbose=1,
                                  patience=5, min_lr=0, mode='auto')
    # Tensorboard 鍩烘湰鍙鍖栥€?
    # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
    #                             batch_size=batch_size, write_graph=True, write_grads=False,
    #                             write_images=False, embeddings_freq=0, embeddings_layer_names=None,
    #                             embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    # 褰撹鐩戞祴鐨勬暟鍊间笉鍐嶆彁鍗囷紝鍒欏仠姝㈣缁冦€?
    EarlyStop = EarlyStopping(monitor='val_mse', min_delta=0.00000000001,
                              patience=epoch * 0.5, verbose=1, mode='auto')

    Check_point = ModelCheckpoint(filepath="./NN/forEpoch/model_{epoch:02d}.weights.h5", save_weights_only=True)



    if retrain:
        NNinput = Input(shape=(taps,))
        # 褰掍竴鍖?
        # NNinput_nor = BatchNormalization()(NNinput)
        NNinput_nor = NNinput
        ## Linear mapping MLP
        # 绾挎€ф槧灏?
        branch1_1 = Dense(int(taps), activation='linear', name='dense1_1')(NNinput_nor)
        # branch1_1_noise = MyLayer(output_dim=int(time_steps), name='noise_linear')(branch1_1)
        branch1_3 = Dense(output_dim, activation='linear', use_bias=False, name='dense1_3')(branch1_1)


        # ## Linear mapping CNN
        # branch1_0 = keras.layers.Reshape((time_steps, 1))(NNinput)
        # branch1_1 = Conv1D(filters=1, kernel_size=32, activation='linear')(branch1_0)
        # branch1_1 = Flatten()(branch1_1)
        # branch1_1_noise = MyLayer(output_dim=int(time_steps), name='noise_linear')(branch1_1)
        # branch1_3 = Dense(output_dim, activation='linear', name='dense1_3')(branch1_1_noise)


        ## Residual mapping
        # NNinput_nor_hollow = Lambda(lambda input: K.concatenate([input[:, 0: int(time_steps/2)], input[:, int(time_steps/2)+1: ]]))(NNinput_nor)
        # branch2_1 = Dense(int(time_steps), activation='relu', name='dense2_1')(NNinput_nor)
        # branch2_1_noise = MyLayer(output_dim=int(time_steps), name='noise_nonlinear')(branch2_1)
        # branch2_2 = Dense(32, activation='tanh', name='dense2_2')(branch2_1_noise)

        # tanh 瀛︿範闈炵嚎鎬х壒鎬? 闈炵嚎鎬ф槧灏?
        branch2_3 = Dense(output_dim, activation='tanh', name='dense2_3')(NNinput_nor)  #'relu'  LeakyReLU(alpha=0.05)  ## tanh
        # Dense鍏ㄨ繛鎺ュ眰


        # attention杈撳嚭涓や釜鏉冮噸
        attention_probs = Dense(2, activation='softmax', name='attention_vec')(NNinput_nor)
        attention_tile_kernel = np.concatenate(
            [
                np.concatenate([np.ones((1, output_dim)), np.zeros((1, output_dim))], axis=1),
                np.concatenate([np.zeros((1, output_dim)), np.ones((1, output_dim))], axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        attention_probs_tile2 = Dense(
            2 * output_dim,
            activation=None,
            use_bias=False,
            trainable=False,
            kernel_initializer=keras.initializers.Constant(attention_tile_kernel),
            name="attention_tile"
        )(attention_probs)

        # 铻嶅悎涓や釜鍒嗘敮
        branch_both = Concatenate()([branch1_3, branch2_3])
        branch_both_mul = Multiply()([branch_both, attention_probs_tile2])  ## 鏉冮噸涓庤緭鍑虹浉涔?
        output = Dense(output_dim, activation=None, use_bias=False, name='Attention',
                       kernel_initializer=keras.initializers.Constant(value=0.5))(branch_both_mul)       # activation=MyLayer(output_dim=output_dim)
        # output = Dense(output_dim, activation=None, use_bias=False, name='output')(branch_both)

        # output = Add()([branch1_3, branch2_3])

        # MyLayer is a noise layer(Gaussian)
        # output_noise = MyLayer(output_dim=output_dim, name='noise')(output)
        # model = Model(inputs=NNinput, outputs=output_noise)
        model = Model(inputs=NNinput, outputs=output)
        atten_model = Model(inputs=NNinput, outputs=attention_probs)

        # plot_model(model, show_shapes=True)
        model.summary()

        model.compile(loss='mse',  # 'mse'  TFLoss
                      optimizer='adam',  #  'adam' 'sgd'   'RMSprop'  'Adagrad' 'Adadelta' 'Adamax'
                      metrics=['mse'])
        HisI = model.fit(x_train, y_train,
                         epochs=epoch,
                         validation_data=(x_test, y_test),
                         batch_size=batch_size,
                         callbacks=[reduce_lr, EarlyStop, Check_point]
                         )  ##  callbacks=[reduce_lr, EarlyStop, Check_point]
        score = model.evaluate(x_test, y_test, batch_size=batch_size)
        model.save(path + 'DNN_' + Extra + '.h5', True, True)
    else:
        # model = keras.models.load_model('DNN3_' + str(36 + i))
        model = keras.models.load_model(path + 'DNN_' + Extra + '.h5',custom_objects={'MyLayer': MyLayer})
        # model._make_predict_function()  # no need to compile
        # HisI=model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
        #         validation_data=(x_test, y_test), callbacks=[reduce_lr,EarlyStop])
        # model.save('DNN_noCP_'+Extra,True,True)

    # predict
    if channelestForPost:
        if retrain:
            predict_Data = data_Rx
        else:
            predict_Data = data_Tx
    else:
        if retrain:
            predict_Data = data_Tx
        else:
            predict_Data = data_Tx  # useless

    data_padding1 = predict_Data[0:int((taps - 1) / 2)]  ## 棰勬祴鍏ㄩ儴娉㈠舰
    data_padding2 = predict_Data[-int((taps - 1) / 2):]
    predict_Data = np.append(data_padding2, predict_Data)
    predict_Data = np.append(predict_Data, data_padding1)
    predict_Data_final = ANN_data_transfer(predict_Data, taps)

    data_equalization = model.predict(predict_Data_final, batch_size=batch_size)
    # data_padding1=x_Rx_2[0:np.int((time_steps-1)/2),:]
    # data_padding2=x_Rx_2[-np.int((time_steps-1)/2): ,:]
    # data_recover=np.row_stack([data_padding1,data_equalization,data_padding2])
    # save data
    data_recover = data_equalization[:, int(output_dim / 2)] #
    data_recover = data_recover.reshape(data_Rx_length, order='C')

    #  Figure
    if 1==1:
        ## 鏋勯€犳í杞?
        m=np.shape(data_recover)[0]
        BaudRate = 50e9
        upsampleno = 4
        xfre = ((np.arange(m) - m / 2) * BaudRate * upsampleno / 1e9) / m
        # Rx after NN
        fft_figure = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data_recover))))
        plt.figure()
        plt.plot(xfre,fft_figure, '-g')
        if channelestForPost:
            plt.title('Rx after NN/Post-equ')
        else:
            plt.title('Tx after NN/Channel')
            fft_figure_error = np.abs(np.fft.fftshift(np.fft.fft(data_Rx)) - np.fft.fftshift(np.fft.fft(data_recover)))
            plt.figure()
            plt.plot(fft_figure_error, '-g')
            plt.title("fft error")

    else:
        # Tx after NN
        fft_figure = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data_recover))))
        plt.figure()
        plt.plot(fft_figure, '-g')
        plt.title('Tx after NN/Pre-equ')

    # Store Model
    newpath = r'''./'''
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # np.savetxt((newpath + '\\' + name_Sa), data_recover, fmt='%1.7e')
    np.savetxt((newpath + name_Sa), data_recover, fmt='%1.7e')
    print('File is ', name_Sa)
    # show noise
    # weight_noise = model.get_layer('noise').get_weights()
    # weight_noise_nonlinear = model.get_layer('noise_nonlinear').get_weights()
    # weight_noise_linear = model.get_layer('noise_linear').get_weights()
    # print('noise stddev is ', weight_noise[0][0])
    # print('noise of nonlinear stddev is ', weight_noise_nonlinear[0][0])
    # print('noise of linear stddev is ', weight_noise_linear[0][0])

    ##  atten(predict_Data_final) (1, 81664, 2)
    if retrain:
        attention_value = atten_model.predict(predict_Data_final, batch_size=batch_size)
        print('attention ', np.mean(attention_value[:, 0]), '  :  ', np.mean(attention_value[:, 1]))

# frequency response of equalized data
# x_fft=20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data_recover))))
# plt.figure(3)
# plt.plot(x_fft,'-k')
# plt.show()
# # model.save('./my_model',True,True)
# # model.save_weights('./weight',True)
if retrain:
    plt.figure()
    np.savetxt('epoch.txt', HisI.epoch, fmt='%1.7e')
    np.savetxt('train_loss.txt', HisI.history['loss'], fmt='%1.7e')
    np.savetxt('val_loss.txt', HisI.history['val_loss'], fmt='%1.7e')
    plt.plot(HisI.epoch, np.log10(HisI.history['loss']), 'r')
    plt.plot(HisI.epoch, np.log10(HisI.history['val_loss']), 'b')
    plt.xlabel('Epoch')
    plt.ylabel('loss(log10)')
    plt.legend(['Train loss ', 'Validation loss of '])

# ## 淇濆瓨棰戣氨鍥?
# m=np.shape(fft_figure)[0]
# np.savetxt('./To_draw/fft_rx.txt',np.abs(np.fft.fftshift(np.fft.fft(data_Rx))))
# np.savetxt('./To_draw/fft_nn.txt', np.abs(np.fft.fftshift(np.fft.fft(data_recover))))
# np.savetxt('./To_draw/fft_mis.txt', np.abs(np.fft.fftshift(np.fft.fft(data_Rx)) - np.fft.fftshift(np.fft.fft(data_recover))))
# np.savetxt('./To_draw/fft_x.txt',xfre)

print("deterministic channel part finished")
if train_gan:
    os.makedirs('./Noise', exist_ok=True)
    ## 鑾峰彇GAN闇€瑕佺殑鍣０鏍锋湰
    rx_simu=np.reshape(data_recover,[-1,1])
    rx_real=np.reshape(data_Rx,[-1,1])
    noise_nor=rx_real-rx_simu  ## 81664 1
    np.savetxt('./Noise/noise.txt',noise_nor)

    plt.figure()
    plt.plot(rx_real[5000:5100], color='lightcoral', alpha=0.5, label='rx')
    plt.plot(rx_simu[5000:5100], color='lightgreen', alpha=0.5, label='out')
    plt.legend()
    plt.title("wave time")

    print("normalization check")
    print("output energy")
    print(np.mean(np.sum(rx_simu ** 2)))
    print("noise energy")
    print(np.mean(np.sum(noise_nor ** 2)))
    print("direct snr ratio")
    print(np.mean(np.sum(rx_simu ** 2)) / np.mean(np.sum(noise_nor ** 2)))
    np.savetxt('./z_zhijiesnr.txt',np.expand_dims(np.mean(np.sum(rx_simu ** 2)) / np.mean(np.sum(noise_nor ** 2)),axis=0))

    print("noise mean and std")
    print(np.mean(noise_nor))
    print(np.sqrt(np.var(noise_nor)))
    plt.figure()
    plt.hist(noise_nor,bins=500,density=True, color='lightgreen')
    plt.title("noise_nor  distribution")

# plt.show()

# to get the weights of each epoch
# weight_flag = np.zeros([epoch, 1])
# for epoch_count in range(epoch):
#     model_temp = model
#     name_weight = './NN/forEpoch/model_' + str(epoch_count+1).zfill(2) +'.hdf5'
#     model_temp.load_weights(filepath=name_weight)
#     weight_flag[epoch_count] = model_temp.get_layer('noise').get_weights()[0][0]
# np.savetxt('weight_flag.txt', weight_flag, fmt='%1.7e')

# flops = get_flops(model)
# print(f"FLOPS: {flops / 10 ** 3:.05} K")
