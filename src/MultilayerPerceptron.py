import numpy as np
import pandas as pd

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
    activator = {
        'ELU': 'elu',
        ''
    }

    def __init__(self, test, train, count_layers, count_neurons, learn_rate, decay, epochs, activator):
        self.dataset_processing(pd.read_csv(test), pd.read_csv(train))
        self.initialisation_of_weight(count_layers, count_neurons)
        self.initialisation_of_b(count_layers, count_neurons)
        self.learn_rate = learn_rate
        self.decay = decay
        self.epochs = epochs
        self.activator = activator

    def dataset_processing(self, train, test):
        train_ = np.array(train)
        test_ = np.array(test)
        self.Y_train = train_[0]
        self.X_train = train_[1:] / 255
        self.Y_test = test_[0]
        self.X_test = test_[1:] / 255

    def initialisation_of_w(self, count_layers, count_neurons):
        count_neurons += [26, 784]
        for i in range(count_layers+1):
            if i+1 != count_layers+1:
                setattr(self, 'w_i_h{}'.format(i + 1), np.random.uniform(-0.5, 0.5, (count_neurons[i], count_neurons[i-1])))
            else:
                self.w_h_o = np.random.uniform(-0.5, 0.5, (count_neurons[i], count_neurons[i-1]))

    def initialisation_of_b(self, count_layers, count_neurons):
        for i in range(count_layers+1):
            if i+1 != count_layers+1:
                setattr(self, 'b_i_h{}'.format(i + 1), np.zeros((count_neurons[i], 1)))
            else:
                self.b_h_o = np.zeros((26, 1))

    def import_activator(self, activator):


    def __feedforward_function(self, img):














class DatasetPreparer:
    _NUM_NEURONS_1 = 196
    _NUM_NEURONS_2 = 49
    _EPOCHS = 3
    _LEARN_RATE = 0.15

    def __init__(self, test='../datasets/emnist-letters/emnist-letters-test.csv',
                 train='../datasets/emnist-letters/emnist-letters-train.csv',count_ne num_neurons_1=196,
                 num_neurons_2=49, epochs=3,
                 ):
        self.__train_label = None
        self.__df_train = pd.read_csv(train)
        self.__one_hot_label_encoding()
        self.__one_hot()
        self.__initialize_weights()

    def __initialize_weights(self):
        self.__w_i_h_1 = np.random.uniform(-0.5, 0.5, (_NUM_NEURONS_1, 784))
        self.__w_i_h_2 = np.random.uniform(-0.5, 0.5, (_NUM_NEURONS_2, _NUM_NEURONS_1))
        self.__w_h_o = np.random.uniform(-0.5, 0.5, (26, _NUM_NEURONS_2))
        self.__b_i_h_1 = np.zeros((_NUM_NEURONS_1, 1))
        self.__b_i_h_2 = np.zeros((_NUM_NEURONS_2, 1))
        self.__b_h_o = np.zeros((26, 1))

    def __one_hot_label_encoding(self):
        self.__train_label = np.array(self.train_label.iloc[:, 0:1]).T
        self.__df_train = np.array(self.df_train.iloc[:, 1:] / 255).T

    @staticmethod
    def __leaky_relu(x, alpha=0.01):
        return x if x > 0 else x*alpha

    @staticmethod
    def __leaky_relu_derivative(x, alpha=0.01):
        return 1 if x > 0 else alpha

    @staticmethod
    def __softmax(z):
        return np.exp(z) / sum(np.exp(z))

    def __feedforward_function(self, img):
        h_pre_1 = self.__b_i_h_1 + self.__w_i_h_1 @ img
        h1 = self.__leaky_relu(h_pre_1)
        h_pre_2 = self.__b_i_h_2 + self.__w_i_h_2 @ h1
        h2 = self.__leaky_relu(h_pre_2)
        o_pre = self.__b_h_o + self.__w_h_o @ h2
        o = self.__softmax(o_pre)
        return h_pre_1, h1, h_pre_2, h2, o_pre, o

    def __one_hot(self):
        y = np.zeros((self.__train_label.size, self.__train_label.max()))
        y[np.arange(self.__train_label.size), self.__train_label] = 1
        self.__train_label = y.T

    def backpropagation(self, h1, h2, o, h_pre_1, h_pre_2, img):
        do = o - self.__train_label
        self.__w_h_o += -_LEARN_RATE * do @ h2.T
        self.__b_h_o += -_LEARN_RATE * do
        dh2 = self.__w_h_o.T @ do
        dh_pre_2 = dh2 * self.__leaky_relu_derivative(h_pre_2)
        self.__w_i_h_2 += -_LEARN_RATE * dh_pre_2 @ h1.T
        self.__b_i_h_2 += -_LEARN_RATE * dh_pre_2
        dh1 = self.__w_i_h_2.T @ dh_pre_2
        dh_pre_1 = dh1 * self.__leaky_relu_derivative(h_pre_1)
        self.__w_i_h_1 += -_LEARN_RATE * dh_pre_1 @ img.T
        self.__b_i_h_1 += -_LEARN_RATE * dh_pre_1



