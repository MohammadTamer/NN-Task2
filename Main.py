import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tkinter as tk
from tkinter import ttk


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    n = len(y_true)
    for i in range(n):
        t = int(y_true[i])
        p = int(y_pred[i])
        cm[t, p] += 1
    return cm


def accuracy_cal(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    matches = 0
    for i in range(n):
        if int(y_true[i]) == int(y_pred[i]):
            matches += 1
    return matches / n


def load_and_preprocess(path, features, label_col):
    df = pd.read_csv(path)
    df = df.fillna(df.mean(numeric_only=True))
    y_encoder = LabelEncoder()
    origin_encoder = LabelEncoder()
    scaler = StandardScaler()
    df['y'] = y_encoder.fit_transform(df[label_col])
    df['OriginLocation'] = origin_encoder.fit_transform(df['OriginLocation'])
    x = scaler.fit_transform(df[features])
    y = df['y'].values
    return x, y


def split_per_class(X, y, train_per_class, test_per_class):
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    for i in classes:
        class_indices = np.where(y == i)[0]
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:train_per_class].tolist())
        test_indices.extend(class_indices[train_per_class:train_per_class + test_per_class].tolist())
    x_train = X[train_indices]
    y_train = y[train_indices]
    x_test = X[test_indices]
    y_test = y[test_indices]
    return x_train, y_train, x_test, y_test


def initialize_layers(neurons_each_level, bias=True):
    params = {}
    for i in range(1, len(neurons_each_level)):
        neurons_now = neurons_each_level[i]
        neurons_pre = neurons_each_level[i - 1]
        lim = np.sqrt(6 / (neurons_pre + neurons_now))
        params[f'weight{i}'] = np.random.uniform(-lim, lim, (neurons_now, neurons_pre))
        if bias:
            params[f'bias{i}'] = np.random.uniform(-lim, lim, (neurons_now, 1))
    return params


def one_hot(y, num_classes=None):
    y = np.array(y).astype(int).ravel()
    if num_classes is None:
        num_classes = np.max(y) + 1
    m = y.shape[0]
    arr = np.zeros((num_classes, m))
    for i, v in enumerate(y):
        arr[v, i] = 1
    return arr


def activation(net, function, a=1.0, b=1.0):
    if function == 'Sigmoid':
        return 1.0 / (1.0 + np.exp(-net))
    elif function == 'Hyperbolic Tangent sigmoid':
        return a * np.tanh(b * net)


def activation_deriv(out, function, a=1.0, b=1.0):
    if function == 'Sigmoid':
        return out * (1 - out)
    elif function == 'Hyperbolic Tangent sigmoid':
        return (a / b) * (a - out) * (a + out)


def forward_propagation(x, parameters, function, bias):
    results = {f'layer_out{0}': x.T}
    if bias:
        layers_count = int(len(parameters) / 2)
    else:
        layers_count = len(parameters)

    for i in range(1, layers_count + 1):
        W = parameters[f'weight{i}']
        out_prev = results[f'layer_out{i - 1}']

        if bias:
            net = W.dot(out_prev) + parameters[f'bias{i}']
        else:
            net = W.dot(out_prev)

        activated_res = activation(net, function)
        results[f'layer_out{i}'] = activated_res
    return results[f'layer_out{layers_count}'], results


def backward_propagation(y_true, parameters, results, function, bias):
    if bias:
        layers_count = int(len(parameters) / 2)
    else:
        layers_count = len(parameters)
    last_output = results[f'layer_out{layers_count}']
    y_one = one_hot(y_true, 3)
    sigmas = {}
    sigma = (y_one - last_output) * activation_deriv(last_output, function)
    sigmas[f'sigma{layers_count}'] = sigma
    for i in reversed(range(1, layers_count)):
        W_next = parameters[f'weight{i + 1}']
        sigma = (W_next.T.dot(sigma)) * activation_deriv(results[f'layer_out{i}'], function)
        sigmas[f'sigma{i}'] = sigma

    return sigmas, results, parameters


def update_parameters(parameters, sigmas, results, lr, bias):
    if bias:
        layers_count = int(len(parameters) / 2)
    else:
        layers_count = len(parameters)

    for i in range(1, layers_count + 1):
        sigma = sigmas[f'sigma{i}']
        prev_out = results[f'layer_out{i - 1}']

        parameters[f'weight{i}'] += lr * sigma.dot(prev_out.T)

        if bias:
            parameters[f'bias{i}'] += lr * np.sum(sigma, axis=1, keepdims=True)


def train_nn(X, y, neurons_each_level, lr, epochs, function, bias=True):
    parameters = initialize_layers(neurons_each_level, bias)
    for i in range(epochs):
        last_output, results = forward_propagation(X, parameters, function, bias)
        sigmas, results, parameters = backward_propagation(y, parameters, results, function, bias)
        update_parameters(parameters, sigmas, results, lr, bias)
    return parameters


def predict(X, parameters, function, bias):
    last_output, _ = forward_propagation(X, parameters, function, bias)
    return np.argmax(last_output, axis=0)


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

    x, y = load_and_preprocess(data_path, features, label_col)

    x_train, y_train, x_test, y_test = split_per_class(x, y, 30, 20)

    all_layers_network = [x_train.shape[1]] + neurons_per_hidden + [len(np.unique(y))]

    parameters = train_nn(x_train, y_train, all_layers_network, lr, epochs, activation_function, bias)

    y_pred = predict(x_test, parameters, activation_function, bias)

    confusion_mat = confusion_matrix(y_test, y_pred, 3)
    accuracy = accuracy_cal(y_test, y_pred)

    results_text.configure(state='normal')
    results_text.delete('1.0', tk.END)
    results_text.insert(tk.END, "Confusion Matrix:\n")
    results_text.insert(tk.END, f"{confusion_mat[0]}\n{confusion_mat[1]}\n{confusion_mat[2]}")
    results_text.insert(tk.END, f"\nOverall Accuracy: {accuracy:.4f}\n")
    results_text.see(tk.END)
    results_text.configure(state='disabled')


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

results_text = tk.Text(control_frame, height=8, width=60, font=("Courier", 10))
results_text.grid(row=7, column=0, columnspan=2, pady=8, padx=4)
results_text.configure(state='normal')

root.mainloop()
