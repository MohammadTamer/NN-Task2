import tkinter as tk
from tkinter import ttk


def generate_back_propagation():
    print("run")


root = tk.Tk()
root.title("Back Propagation")
control_frame = tk.Frame(root)
root.geometry("700x300")
control_frame.pack()

hidden_layer_no_label = tk.Label(control_frame, text="Number Of Hidden Layer", bg="#f0f0f0", font=("Helvetica", 12))
hidden_layer_no_label.grid(row=0, column=0)
hidden_layer_no_entry = tk.Entry(control_frame, font=("Helvetica", 12))
hidden_layer_no_entry.grid(row=0, column=1)

neurons_per_level_label = tk.Label(control_frame, text="Neurons In Each Hidden Layer", bg="#f0f0f0",
                                   font=("Helvetica", 12))
neurons_per_level_label.grid(row=1, column=0)
neurons_per_level_entry = tk.Entry(control_frame, font=("Helvetica", 12))
neurons_per_level_entry.grid(row=1, column=1)

learning_rate_label = tk.Label(control_frame, text="Learning Rate", bg="#f0f0f0", font=("Helvetica", 12))
learning_rate_label.grid(row=2, column=0)
learning_rate_entry = tk.Entry(control_frame, font=("Helvetica", 12))
learning_rate_entry.grid(row=2, column=1)

epochs_label = tk.Label(control_frame, text="Epochs", bg="#f0f0f0", font=("Helvetica", 12))
epochs_label.grid(row=3, column=0)
epochs_entry = tk.Entry(control_frame, font=("Helvetica", 12))
epochs_entry.grid(row=3, column=1)

bias_choice = tk.BooleanVar()
bias_checkbox = tk.Checkbutton(control_frame, text="Bias", variable=bias_choice, font=("Helvetica", 12))
bias_checkbox.grid(row=4, column=0)

activation_function_label = tk.Label(control_frame, text="Activation Function", bg="#f0f0f0", font=("Helvetica", 12))
activation_function_label.grid(row=5, column=0)

functions = ["Sigmoid", "Hyperbolic Tangent sigmoid"]

activation_function_combobox = ttk.Combobox(control_frame, values=functions, state="readonly", font=("Helvetica", 12))
activation_function_combobox.grid(row=5, column=1)
activation_function_combobox.set(value="Sigmoid")

generate_button = tk.Button(control_frame, text='Clear', command=generate_back_propagation, bg='red',
                            font=("Helvetica", 12), width=25)
generate_button.grid(row=6, column=1, pady=5)

root.mainloop()
