import sys

import numpy as np
import pandas as pd
import Activations

# test = '../datasets/emnist-letters/emnist-letters-test.csv'
# train = '../datasets/emnist-letters/emnist-letters-train.csv'


# lr decay
# how often
# test dataset, train dataset, количество скрытых слоев, сколько нейронов в каждом слое,
# лр, decay, количество эпох, выбор активатора

#логирование обучения (на выходе тебе нужно посчитать Recall, Fmeasure, Accuracy and Precision)
#необязательно(для каждой эпохи - MSE, сколько он угадал, время) в текстовик
#сохранение модели в текстовик - сколько слоев, какого размера слои, все веса

#по рофлу: оптимизаторы, l1/l2/dropout,
#KingScheduler
class MLP:

    def __init__(self, test, train, count_layers, count_neurons, learn_rate, decay, epochs, activator):
        self.dataset_processing(pd.read_csv(test), pd.read_csv(train))
        self.w = self.__initialisation_of_w(count_layers, count_neurons)
        self.b = self.__initialisation_of_b(count_layers, count_neurons)
        self.act_fn, self.act_fn_grad = self.__import_activator(activator)

        self.learn_rate = learn_rate
        self.decay = decay
        self.epochs = epochs

    def dataset_processing(self, train, test):
        train_ = np.array(train).T
        test_ = np.array(test).T
        y_train = np.zeros((train_[0].size,  train_[0].max()))
        y_train[np.arange(train_[0].size),  train_[0]] = 1
        y_test = np.zeros((test_[0].size, test_[0].max()))
        y_test[np.arange(test_[0].size), test_[0]] = 1
        self.Y_train = y_train.T
        self.X_train = train_[1:] / 255
        self.Y_test = y_test.T
        self.X_test = test_[1:] / 255


    @staticmethod
    def __initialisation_of_w(count_layers, count_neurons):
        count_neurons += [26, 784]
        w = []
        for i in range(count_layers+1):
            w.append(np.random.uniform(-0.5, 0.5, (count_neurons[i], count_neurons[i-1])))
        return w

    @staticmethod
    def __initialisation_of_b(count_layers, count_neurons):
        b = []
        for i in range(count_layers+1):
            b.append(np.zeros((count_neurons[i], 1)))
        return b

    @staticmethod
    def __import_activator(activator):
        act_fn = getattr(sys.modules['Activations'], activator)
        act_fn_grad = getattr(sys.modules['Activations'], activator + '_grad')
        return act_fn, act_fn_grad

    @staticmethod
    def __softmax(z):
        return np.exp(z) / sum(np.exp(z))

    def __feedforward_function(self, count_layers, img):
        h = [img]
        h_pre = []
        for layer in count_layers + 1:
            h_pre.append(self.b[layer] + self.w[layer @ h[-1]])
            h.append(self.act_fn(h_pre[-1]))
        h[-1] = self.__softmax(h_pre.pop())
        return h, h_pre

    def __backpropagation(self, count_layers, h, h_pre):
        dh = 0
        dh_pre = 0
        for layer in count_layers + 1:
            if layer == 0:
                dh = h.pop() - self.Y_train
                self.w[count_layers - layer] += -self.learn_rate * dh @ h.pop()
                self.b[count_layers - layer] += -self.learn_rate * dh
            else:
                dh = self.w[-1].T @ dh if layer == 1 else self.w[-layer] @ dh_pre
                dh_pre = dh * self.act_fn_grad(h_pre.pop())
                self.w[count_layers - layer] += -self.learn_rate * dh_pre @ h.pop()
                self.b[count_layers - layer] += -self.learn_rate * dh_pre






        # do = o - self.__train_label
        # self.__w_h_o += -_LEARN_RATE * do @ h2.T
        # self.__b_h_o += -_LEARN_RATE * do
        # dh2 = self.__w_h_o.T @ do
        # dh_pre_2 = dh2 * self.__leaky_relu_derivative(h_pre_2)
        # self.__w_i_h_2 += -_LEARN_RATE * dh_pre_2 @ h1.T
        # self.__b_i_h_2 += -_LEARN_RATE * dh_pre_2
        # dh1 = self.__w_i_h_2.T @ dh_pre_2
        # dh_pre_1 = dh1 * self.__leaky_relu_derivative(h_pre_1)
        # self.__w_i_h_1 += -_LEARN_RATE * dh_pre_1 @ img.T
        # self.__b_i_h_1 += -_LEARN_RATE * dh_pre_1

















#
# class DatasetPreparer:
#     _NUM_NEURONS_1 = 196
#     _NUM_NEURONS_2 = 49
#     _EPOCHS = 3
#     _LEARN_RATE = 0.15
#
#     def __init__(self, test='../datasets/emnist-letters/emnist-letters-test.csv',
#                  train='../datasets/emnist-letters/emnist-letters-train.csv',count_ne num_neurons_1=196,
#                  num_neurons_2=49, epochs=3,
#                  ):
#         self.__train_label = None
#         self.__df_train = pd.read_csv(train)
#         self.__one_hot_label_encoding()
#         self.__one_hot()
#         self.__initialize_weights()
#
#     def __initialize_weights(self):
#         self.__w_i_h_1 = np.random.uniform(-0.5, 0.5, (_NUM_NEURONS_1, 784))
#         self.__w_i_h_2 = np.random.uniform(-0.5, 0.5, (_NUM_NEURONS_2, _NUM_NEURONS_1))
#         self.__w_h_o = np.random.uniform(-0.5, 0.5, (26, _NUM_NEURONS_2))
#         self.__b_i_h_1 = np.zeros((_NUM_NEURONS_1, 1))
#         self.__b_i_h_2 = np.zeros((_NUM_NEURONS_2, 1))
#         self.__b_h_o = np.zeros((26, 1))
#
#     def __one_hot_label_encoding(self):
#         self.__train_label = np.array(self.train_label.iloc[:, 0:1]).T
#         self.__df_train = np.array(self.df_train.iloc[:, 1:] / 255).T
#
#     @staticmethod
#     def __leaky_relu(x, alpha=0.01):
#         return x if x > 0 else x*alpha
#
#     @staticmethod
#     def __leaky_relu_derivative(x, alpha=0.01):
#         return 1 if x > 0 else alpha
#
#     @staticmethod
#     def __softmax(z):
#         return np.exp(z) / sum(np.exp(z))
#
#     def __feedforward_function(self, img):
#         h_pre_1 = self.__b_i_h_1 + self.__w_i_h_1 @ img
#         h1 = self.__leaky_relu(h_pre_1)
#         h_pre_2 = self.__b_i_h_2 + self.__w_i_h_2 @ h1
#         h2 = self.__leaky_relu(h_pre_2)
#         o_pre = self.__b_h_o + self.__w_h_o @ h2
#         o = self.__softmax(o_pre)
#         return h_pre_1, h1, h_pre_2, h2, o_pre, o
#
#     def __one_hot(self):
#         y = np.zeros((self.__train_label.size, self.__train_label.max()))
#         y[np.arange(self.__train_label.size), self.__train_label] = 1
#         self.__train_label = y.T
#
#     def backpropagation(self, h1, h2, o, h_pre_1, h_pre_2, img):
#         do = o - self.__train_label
#         self.__w_h_o += -_LEARN_RATE * do @ h2.T
#         self.__b_h_o += -_LEARN_RATE * do
#         dh2 = self.__w_h_o.T @ do
#         dh_pre_2 = dh2 * self.__leaky_relu_derivative(h_pre_2)
#         self.__w_i_h_2 += -_LEARN_RATE * dh_pre_2 @ h1.T
#         self.__b_i_h_2 += -_LEARN_RATE * dh_pre_2
#         dh1 = self.__w_i_h_2.T @ dh_pre_2
#         dh_pre_1 = dh1 * self.__leaky_relu_derivative(h_pre_1)
#         self.__w_i_h_1 += -_LEARN_RATE * dh_pre_1 @ img.T
#         self.__b_i_h_1 += -_LEARN_RATE * dh_pre_1
#
#
#
