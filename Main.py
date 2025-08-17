import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tkinter as tk
from tkinter import ttk


def load_and_preprocess(path, features, label_col):
    df = pd.read_csv(path)
    df = df.dropna()
    y_encoder = LabelEncoder()
    origin_encoder = LabelEncoder()
    scaler = StandardScaler()
    df['y'] = y_encoder.fit_transform(df[label_col])
    df['OriginLocation'] = origin_encoder.fit_transform(df['OriginLocation'])
    X = scaler.fit_transform(df[features])
    y = df['y'].values
    return X, y


def split_per_class(X, y, train_per_class, test_per_class):
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    for i in classes:
        class_indices = np.where(y == i)[0]
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:train_per_class].tolist())
        remaining_indices = class_indices.size - train_per_class
        test_indices.extend(class_indices[train_per_class:train_per_class + remaining_indices].tolist())
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test


def initialize_layers(neurons_each_level, bias=True):
    params = {}
    for i in range(1, len(neurons_each_level)):
        n_l = neurons_each_level[i]
        n_prev = neurons_each_level[i - 1]
        # Xavier/Glorot uniform
        limit = np.sqrt(6.0 / (n_prev + n_l))
        W = np.random.uniform(-limit, limit, size=(n_l, n_prev))
        params[f'weight{i}'] = W
        if bias:
            params[f'bias{i}'] = np.zeros((n_l, 1))
    return params


def activation(net, function, a=1.0, b=1.0):
    if function == 'Sigmoid':
        return 1.0 / (1.0 + np.exp(-net))
    elif function == 'Hyperbolic Tangent sigmoid':
        return a * np.tanh(b * net)


def activation_deriv(net, function='Sigmoid', a=1.0, b=1.0):
    if function == 'Sigmoid':
        return activation(net, function) * (1 - activation(net, function))
    elif function == 'Hyperbolic Tangent sigmoid':
        return (a / b) * (a - activation(net, function)) * (a + activation(net, function))


def softmax(net):
    res = np.exp(net - np.max(net, axis=0, keepdims=True))
    return res / np.sum(res, axis=0, keepdims=True)


def forward_propagation(X, parameters, function, bias):
    results = {f'layer{0}': X.T}
    if bias:
        weight_counts = int(len(parameters) / 2)
    else:
        weight_counts = len(parameters)

    for i in range(1, weight_counts + 1):
        W = parameters[f'weight{i}']
        A_prev = results[f'layer{i - 1}']

        if bias:
            net = W.dot(A_prev) + parameters[f'bias{i}']
        else:
            net = W.dot(A_prev)

        if i < weight_counts:
            activated = activation(net, function)
        else:
            activated = softmax(net)
        results[f'layer{i}'] = activated
    return results[f'layer{weight_counts}'], results


def backward_propagation(y_true, parameters, results, function, bias):
    grads = {}
    m = results['layer0'].shape[1]
    L = sum(1 for k in parameters if k.startswith('w'))
    # determine number of output classes from last W
    output_neurons = parameters[f'weight{L}'].shape[0]
    # one-hot encode
    Y = np.eye(output_neurons)[y_true].T
    A_L = results[f'layer{L}']
    # derivative for softmax + cross-entropy
    dA = A_L - Y
    for l in reversed(range(1, L + 1)):
        A_prev = results[f'layer{l - 1}']
        dW = (1 / m) * dA.dot(A_prev.T)
        db = (1 / m) * np.sum(dA, axis=1, keepdims=True) if bias else None
        grads[f'dW{l}'], grads[f'db{l}'] = dW, db
        if l > 1:
            dA = parameters[f'weight{l}'].T.dot(dA) * activation_deriv(results[f'layer{l - 1}'], function)
    return grads


def update_parameters(parameters, grads, lr, bias):
    L = sum(1 for k in parameters if k.startswith('w'))
    for l in range(1, L + 1):
        parameters[f'weight{l}'] -= lr * grads[f'dW{l}']
        if bias:
            parameters[f'bias{l}'] -= lr * grads[f'db{l}']


def train_nn(X, y, neurons_each_level, lr, epochs, function='Sigmoid', bias=True):
    parameters = initialize_layers(neurons_each_level, bias)
    for i in range(epochs):
        AL, results = forward_propagation(X, parameters, function, bias)
        grads = backward_propagation(y, parameters, results, function, bias)
        update_parameters(parameters, grads, lr, bias)
    return parameters


def predict(X, parameters, function, bias):
    AL, _ = forward_propagation(X, parameters, function, bias)
    return np.argmax(AL, axis=0)


def main():
    data_path = 'penguins.csv'
    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
    label_col = 'Species'

    num_hidden_layers = int(hidden_layer_no_entry.get())
    splitted = neurons_per_level_entry.get().split(',')
    neurons_per_hidden = []
    for i in range(num_hidden_layers):
        neurons_per_hidden.append(int(splitted[i]))
    lr = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())
    activation_function = str(activation_function_combobox.get())
    bias = bool(bias_choice.get())

    X, y = load_and_preprocess(data_path, features, label_col)

    X_train, y_train, X_test, y_test = split_per_class(X, y, 30, 20)

    all_layers_network = [X_train.shape[1]] + neurons_per_hidden + [len(np.unique(y))]

    parameters = train_nn(X_train, y_train, all_layers_network, lr, epochs, activation_function, bias)

    y_pred = predict(X_test, parameters, activation_function, bias)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print('Confusion Matrix:')
    print(cm)
    print(f'Overall Accuracy: {acc:.4f}')


root = tk.Tk()
root.title("Back Propagation")
control_frame = tk.Frame(root)
root.geometry("700x300")
control_frame.pack()

hidden_layer_no_label = tk.Label(control_frame, text="Number Of Hidden Layer", bg="#f0f0f0", font=("Helvetica", 12))
hidden_layer_no_label.grid(row=0, column=0)
hidden_layer_no_entry = tk.Entry(control_frame, font=("Helvetica", 12))
hidden_layer_no_entry.grid(row=0, column=1)
hidden_layer_no_entry.insert(0, '2')

neurons_per_level_label = tk.Label(control_frame, text="Neurons In Each Hidden Layer", bg="#f0f0f0",
                                   font=("Helvetica", 12))
neurons_per_level_label.grid(row=1, column=0)
neurons_per_level_entry = tk.Entry(control_frame, font=("Helvetica", 12))
neurons_per_level_entry.grid(row=1, column=1)
neurons_per_level_entry.insert(0, '3,4')

learning_rate_label = tk.Label(control_frame, text="Learning Rate", bg="#f0f0f0", font=("Helvetica", 12))
learning_rate_label.grid(row=2, column=0)
learning_rate_entry = tk.Entry(control_frame, font=("Helvetica", 12))
learning_rate_entry.grid(row=2, column=1)
learning_rate_entry.insert(0, '0.01')

epochs_label = tk.Label(control_frame, text="Epochs", bg="#f0f0f0", font=("Helvetica", 12))
epochs_label.grid(row=3, column=0)
epochs_entry = tk.Entry(control_frame, font=("Helvetica", 12))
epochs_entry.grid(row=3, column=1)
epochs_entry.insert(0, '1000')

bias_choice = tk.BooleanVar()
bias_checkbox = tk.Checkbutton(control_frame, text="Bias", variable=bias_choice, font=("Helvetica", 12))
bias_checkbox.grid(row=4, column=0)

activation_function_label = tk.Label(control_frame, text="Activation Function", bg="#f0f0f0",
                                     font=("Helvetica", 12))
activation_function_label.grid(row=5, column=0)

functions = ["Sigmoid", "Hyperbolic Tangent sigmoid"]

activation_function_combobox = ttk.Combobox(control_frame, values=functions, state="readonly",
                                            font=("Helvetica", 12))
activation_function_combobox.grid(row=5, column=1)
activation_function_combobox.set(value="Sigmoid")

generate_button = tk.Button(control_frame, text='Run', command=main, bg='red',
                            font=("Helvetica", 12), width=25)
generate_button.grid(row=6, column=1, pady=5)

root.mainloop()
