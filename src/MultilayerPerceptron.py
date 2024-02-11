import sys

import numpy as np
import pandas as pd
import Activations
import DropOut
import Losses
import BatchNormalization


# test = '../datasets/emnist-letters/emnist-letters-test.csv'
# train = '../datasets/emnist-letters/emnist-letters-train.csv'


# lr decay
# how often
# test dataset, train dataset, количество скрытых слоев, сколько нейронов в каждом слое,
# лр, decay, количество эпох, выбор активатора

# логирование обучения (на выходе тебе нужно посчитать Recall, Fmeasure, Accuracy and Precision)
# необязательно(для каждой эпохи - MSE, сколько он угадал, время) в текстовик
# сохранение модели в текстовик - сколько слоев, какого размера слои, все веса

# по рофлу: оптимизаторы, l1/l2/dropout,
# KingScheduler
class MLP:

    def __init__(self, test, train, count_layers, count_neurons, learn_rate, decay, epochs, activator, dropout,
                 dropout_rate):
        self.dataset_processing(pd.read_csv(test), pd.read_csv(train))
        self.w = self.__initialisation_of_w(count_layers, count_neurons)
        self.b = self.__initialisation_of_b(count_layers, count_neurons)
        self.act_fn, self.act_fn_grad = self.__import_activator(activator)
        self.dropout = self.__import_dropout(dropout)

        self.bn = BatchNormalization.BN(count_neurons)

        self.learn_rate = learn_rate
        self.decay = decay
        self.epochs = epochs
        self.dropout_rate = dropout_rate

    def dataset_processing(self, train, test):
        train_ = np.array(train).T
        test_ = np.array(test).T
        y_train = np.zeros((train_[0].size, train_[0].max()))
        y_train[np.arange(train_[0].size), train_[0]] = 1
        y_test = np.zeros((test_[0].size, test_[0].max()))
        y_test[np.arange(test_[0].size), test_[0]] = 1
        self.Y_train = y_train.T
        self.X_train = train_[1:] / 255
        self.Y_test = y_test.T
        self.X_test = test_[1:] / 255

    @staticmethod
    def __import_dropout(dropout):
        return getattr(sys.modules['DropOut'], dropout)

    @staticmethod
    def __initialisation_of_w(count_layers, count_neurons):
        count_neurons += [26, 784]
        w = []
        for i in range(count_layers + 1):
            w.append(np.random.uniform(-0.5, 0.5, (count_neurons[i], count_neurons[i - 1])))
        return w

    @staticmethod
    def __initialisation_of_b(count_layers, count_neurons):
        b = []
        for i in range(count_layers + 1):
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

    def __feedforward(self, count_layers, img):
        h = [img]
        h_pre = []
        for layer in range(count_layers + 1):
            h_pre.append(self.b[layer] + self.w[layer] @ h[-1])
            h_pre[-1] = h_pre[-1] if layer == 0 and layer == count_layers else self.bn.feed_bn(h_pre[-1], layer)
            h.append(self.act_fn(h_pre[-1]))
            if layer < count_layers:
                h[-1] = self.dropout(h[-1], self.dropout_rate)
        h[-1] = self.__softmax(h_pre.pop())
        return h, h_pre

    def __backpropagation(self, count_layers, h, h_pre):
        dh = 0
        dh_pre = 0
        loss = Losses.cross_entropy(self.Y_train, h[-1])
        for layer in range(count_layers + 1):
            if layer == 0:
                dh = Losses.cross_entropy_grad(self.Y_train, h.pop())
                self.w[count_layers - layer] += -self.learn_rate * dh @ h.pop()
                self.b[count_layers - layer] += -self.learn_rate * dh
            else:
                dh = self.w[-1].T @ dh if layer == 1 else self.w[-layer] @ dh_pre
                h_pre_ = h_pre.pop()
                dh_pre = dh * self.act_fn_grad(h_pre_)
                self.w[count_layers - layer] += -self.learn_rate * dh_pre @ h.pop()
                self.b[count_layers - layer] += -self.learn_rate * dh_pre
                if layer != count_layers:
                    self.bn.back_bn(dh_pre, self.learn_rate, h_pre_, layer)
        return loss

    # def train(self):
