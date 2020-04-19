from django.shortcuts import render
import numpy as np
from neural.NeuralNetwork import NeuralNetwork
from django.conf import settings
import os
import json
from django.http import HttpResponse


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
neurals = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open(os.path.join(settings.STATIC_DIR, 'mnist_train.csv'))
training_data_list = training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    neurals.train(inputs, targets)

out = []


def getNumberWithPage(request):
    l = [float(i) for i in request.GET['pixs'].replace(' ', '').replace('[', '').replace(']', '').split(',')]
    inputs = (np.asfarray(l) / 255.0 * 0.99) + 0.01
    outputs = neurals.query(inputs)
    out.append([round(i, 2) for i in outputs.flatten() * 100])
    return render(request, 'index.html',
                  {'number': out, 'myNum': np.argmax(outputs)})


def getNumber(request):
    l = [float(i) for i in request.GET['pixs'].replace(' ', '').replace('[', '').replace(']', '').split(',')]
    inputs = (np.asfarray(l) / 255.0 * 0.99) + 0.01
    outputs = neurals.query(inputs)

    return HttpResponse(json.dumps({'num': int(np.argmax(outputs))}), content_type="application/json")


def main(request):
    return render(request, 'index.html', {})
