import math
import sys
import numpy as np
import pandas as pd

inputNum = 12
neuron_layer_1 = 47
neuron_layer_2 = 60
outputNum = 2
lines_count_train = 79
lines_count_test = 10

class NN(object):
    def __init__(self, learning_rate=0.1):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (neuron_layer_1, inputNum))
        self.weights_1_2 = np.random.normal(0.0, 1, (neuron_layer_2, neuron_layer_1))
        self.weights_2_3 = np.random.normal(0.0, 2 ** 0.5, (outputNum, neuron_layer_2))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, arr):
        inputs = []
        for i in range(0, inputNum):
            if arr[i] is None:
                inputs.append(0.0)
            else:
                inputs.append(arr[i])
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)

        inputs_3 = np.dot(self.weights_2_3, outputs_2)
        outputs_3 = self.sigmoid_mapper(inputs_3)
        return outputs_3

    def train(self, arr, exp, isKGF=True):
        inputs = np.zeros(inputNum)
        for i in range(0, inputNum):
            if arr[i] is None:
                inputs[i] = 0.0
            else:
                inputs[i] = arr[i]
        if isKGF:
            i = 1
        else:
            i = 0
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)

        inputs_3 = np.dot(self.weights_2_3, outputs_2)
        outputs_3 = self.sigmoid_mapper(inputs_3)
        actual_predict = outputs_3[i]
        expected_predict = exp[i]
        error_layer_3 = (np.array([actual_predict - expected_predict]))
        gradient_layer_3 = actual_predict * (1 - actual_predict)
        weights_delta_layer_3 = error_layer_3 * gradient_layer_3
        self.weights_2_3[i] -= (np.dot(weights_delta_layer_3, outputs_2.reshape(1, len(outputs_2)))) * self.learning_rate

        error_layer_2 = weights_delta_layer_3 * self.weights_2_3[i]
        gradient_layer_2 = np.dot(outputs_2, (1 - outputs_2))
        #хз верно ли
        weights_delta_layer_2 = np.dot(error_layer_2, gradient_layer_2) + np.dot(weights_delta_layer_3, outputs_2.reshape(1, len(outputs_2)))
        self.weights_1_2 -= (np.dot(weights_delta_layer_2.reshape(len(weights_delta_layer_2), 1),
                                    outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = np.dot(weights_delta_layer_2.reshape(1, len(weights_delta_layer_2)), self.weights_1_2)
        gradient_layer_1 = np.dot(outputs_1, (1 - outputs_1))
        partDelta = np.dot(error_layer_1, gradient_layer_1) #то, что получили на этом слое
        errPrevLayer = np.dot(weights_delta_layer_2.reshape(len(weights_delta_layer_2), 1), outputs_1.reshape(1, len(outputs_1)))
        errLayers = np.sum(errPrevLayer, axis=0)
        weights_delta_layer_1 = partDelta + errLayers
        #опять хз
        #weights_delta_layer_1 = dott + (np.dot(weights_delta_layer_2.reshape(len(weights_delta_layer_2), 1), outputs_1.reshape(1, len(outputs_1))))
        for k in range(0, neuron_layer_1):
            for j in range (0, inputNum):
                if arr[i] is None:
                    continue
                else:
                    self.weights_0_1[k, j] -= (inputs[j] * weights_delta_layer_1[0][k] * self.learning_rate[0])


def mse(y, Y):
    return np.mean((y - Y) ** 2)


epochs = 100
lr = 0.001

network = NN(learning_rate=lr)
table = pd.read_csv(r'C:\Users\Ekaterina\PycharmProjects\untitled2\train.csv', sep=";", header=[0], encoding="windows-1251")
table = table.replace(',', '.', regex=True)
for i, row in table.iterrows():
    for column in table.columns:
        try:
            table.at[i, column] = float(table.at[i, column])
        except ValueError:
            table.at[i, column] = None
        if math.isnan(table.at[i, column]):
            table.at[i, column] = None


for e in range(epochs):
    inputs_kgf = []
    correct_predictions_kgf = []
    predictions_kgf = []
    for i, row in table.iterrows():
        correct_predict = []
        correct_predict.append(table.values.tolist()[i][inputNum])  # G_total
        correct_predict.append(table.values.tolist()[i][inputNum + 1])  # КГФ
        input_stat = np.array(table.values.tolist()[i])
        network.train(np.array(input_stat), correct_predict)
        inputs_kgf.append(np.array(input_stat))
        correct_predictions_kgf.append(correct_predict)

    for i in range(0, lines_count_train):
        prediction_kgf = network.predict(np.array(inputs_kgf[i]))
        predictions_kgf.append(prediction_kgf[1])
    train_loss_kgf = mse(np.array(predictions_kgf), np.array(correct_predictions_kgf).T[1])
    # конец КГФ

    predictions_g_total = []
    correct_predictions_g_total = []
    inputs_g_total = []
    for i, row in table.iterrows():
        correct_predict = []
        g_total = table.values.tolist()[i][inputNum]    # G_total
        if g_total is None:
            continue
        correct_predict.append(table.values.tolist()[i][inputNum + 1])  # КГФ
        correct_predict.append(g_total)
        input_stat = np.array(table.values.tolist()[i])
        network.train(np.array(input_stat), correct_predict, False)
        inputs_g_total.append(np.array(input_stat))
        correct_predictions_g_total.append(correct_predict)

    k = 0
    for i in range(0, lines_count_train):
        if table.values.tolist()[i][inputNum] is None:
            continue
        prediction_g_total = network.predict(np.array(inputs_g_total[k]))
        k = k + 1
        predictions_g_total.append(prediction_g_total[0])
    train_loss_g_total = mse(np.array(predictions_g_total), np.array(correct_predictions_g_total).T[0])
    # train_loss = mse(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write(
        "\nProgress: {}, Training loss KGF: {}, Training loss G_total: {}".format(str(100 * e / float(epochs))[:4], str(train_loss_kgf)[:5], str(train_loss_g_total)[:5]))
    # конец g_total

# начало проверки на тестовом сете
test_table = pd.read_csv(r'C:\Users\Ekaterina\PycharmProjects\untitled2\test.csv', sep=";", header=[0],
                    encoding="windows-1251")
test_table = test_table.replace(',', '.', regex=True)
for i, row in test_table.iterrows():
    for column in test_table.columns:
        try:
            test_table.at[i, column] = float(test_table.at[i, column])
        except ValueError:
            test_table.at[i, column] = None
        if math.isnan(test_table.at[i, column]):
            test_table.at[i, column] = None

correct_predictions = []
predictions = []
predict_arr_kgf = []
correct_arr_kgf = []
predict_arr_g_total = []
correct_arr_g_total = []
for i in range(0, lines_count_test):
    input = np.array(table.values.tolist()[i])
    prediction = network.predict(np.array(input))
    predictions.append(prediction[1])
    correct_g_total = table.values.tolist()[i][inputNum]  # G_total
    correct_kgf = table.values.tolist()[i][inputNum + 1]  # КГФ
    g_total_predict = prediction[0]
    kgf_predict = prediction[1]
    sys.stdout.write(
        "\nTest #{}: Expected KGF: {}, Predicted KGF: {}, Err: {}".format(str(i), str(correct_kgf)[:5], str(kgf_predict)[:5], str(mse(kgf_predict, correct_kgf))))
    correct_arr_kgf.append(correct_kgf)
    predict_arr_kgf.append(kgf_predict)
    if correct_g_total is None:
        continue
    else:
        sys.stdout.write(
            "\nTest #{}: Expected G_total: {}, Predicted G_total: {}, Err: {}".format(str(i), str(correct_g_total)[:5],
                                                                     str(g_total_predict)[:5], str(mse(g_total_predict, correct_g_total))))
    correct_arr_g_total.append(correct_g_total)
    predict_arr_g_total.append(g_total_predict)

sys.stdout.write("\nKGF err: {}, G_total err: {}".format(str(mse(np.array(correct_arr_kgf), np.array(predict_arr_kgf))), str(mse(np.array(correct_arr_g_total), np.array(predict_arr_g_total)))))
