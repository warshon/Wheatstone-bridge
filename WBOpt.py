"""
Created on Sun Sep 15 14:41:35 2024

@author: 彭泽彦
"""
from skopt import gp_minimize
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys
import os
import threading
import queue

progress_queue = queue.Queue()
Ig_cache = {}
is_calculating = False  # Global variable to track the calculation status

# Define the current function
def IG_function(Rx, R2, R3, Rb, rg, E, h):
    R4 = Rx * R3 / R2 + h
    D = R4 * Rb * rg + R3 * Rb * (R4 + rg) + (R4 + Rb) * rg * Rx + R3 * (R4 + Rb + rg) * Rx + \
        R2 * (Rb * (rg + Rx) + R3 * (R4 + rg + Rx) + R4 * (Rb + rg + Rx))
    Ig_plus = E * (R2 * R4 - Rx * R3) / D

    R4 = Rx * R3 / R2 - h
    D = R4 * Rb * rg + R3 * Rb * (R4 + rg) + (R4 + Rb) * rg * Rx + R3 * (R4 + Rb + rg) * Rx + \
        R2 * (Rb * (rg + Rx) + R3 * (R4 + rg + Rx) + R4 * (Rb + rg + Rx))
    Ig_minus = E * (R2 * R4 - Rx * R3) / D

    return Ig_plus, Ig_minus

def sensitivity_function(Rx, R2, R3, Rb, rg, E, h):
    params = (R2, R3)
    if params in Ig_cache:
        Ig_plus_h = Ig_cache[params]['plus_h']
        Ig_minus_h = Ig_cache[params]['minus_h']
    else:
        Ig_plus_h, Ig_minus_h = IG_function(Rx, R2, R3, Rb, rg, E, h)
        Ig_cache[params] = {'plus_h': Ig_plus_h, 'minus_h': Ig_minus_h}

    return (Rx * R3 / R2) * (Ig_plus_h - Ig_minus_h) / (2 * h)

def objective(params, Rx, Rb, rg, E):
    R2, R3 = params
    return -sensitivity_function(Rx, R2, R3, Rb, rg, E, 0.0005)

def search_around_best_values(Rx, Rb, rg, E, update_progress, n):
    space = [(0.01, Rx), (0.01, Rx)]
    
    result = gp_minimize(lambda params: objective(params, Rx, Rb, rg, E), space, 
                          n_calls=n, n_initial_points=5, random_state=None, 
                           callback=[update_progress])
    
    best_R2, best_R3 = result.x
    max_sensitivity = -result.fun
    best_values = [Rx, round(best_R2, 1), round(best_R3, 1), 
                   round(Rx * best_R3 / best_R2, 1), round(max_sensitivity, 8), 
                   round(best_R2 / best_R3, 2)]
    return best_values

# Create main window
root = tk.Tk()
root.title("Wheatstone Bridge Sensitivity Optimization")
root.geometry("600x450+300+100")

# Set row and column weights
root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(1, weight=0)
root.grid_rowconfigure(2, weight=0)
root.grid_rowconfigure(7, weight=1)
root.grid_rowconfigure(9, weight=0)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

# Create label controls
labels = ["E (V):", "rb (Ω):", "rg (Ω):", "Rx (Ω):"]
for i, text in enumerate(labels):
    label = tk.Label(root, text=text, font=("Times New Roman", 14), fg="blue")
    label.grid(row=i, column=0, padx=10, pady=3, sticky='e')

# Create entry controls
e_entry = tk.Entry(root, font=("Times New Roman", 14))
e_entry.grid(row=0, column=1, padx=10, pady=3, sticky='ew')

rb_entry = tk.Entry(root, font=("Times New Roman", 14))
rb_entry.grid(row=1, column=1, padx=10, pady=3, sticky='ew')

rg_entry = tk.Entry(root, font=("Times New Roman", 14))
rg_entry.grid(row=2, column=1, padx=10, pady=3, sticky='ew')

rx_entry = tk.Entry(root, font=("Times New Roman", 14))
rx_entry.grid(row=3, column=1, padx=10, pady=3, sticky='ew')

# Create result label control
result_label = tk.Label(root, text="", font=("Times New Roman", 15))
result_label.grid(row=6, column=1, columnspan=3, padx=1, pady=4, sticky='w')

# Create progress bar control
progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=250, mode='determinate')
progress.grid(row=9, column=0, columnspan=3, padx=5, pady=0)

# Create calculation button click event handler function
def update_gui():
    global is_calculating
    try:
        best_values = progress_queue.get_nowait()
        result_label.config(
            text=(
                f"                     Best R2:     {best_values[1]:10.1f}Ω\n"
                f"                     Best R3:      {best_values[2]:10.1f}Ω\n"
                f"               Approx. R4:      {best_values[3]:10.1f}Ω\n"
                f"                 Sensitivity:       {best_values[4]:10.2e}\n"
                f"                       R2/R3:        {best_values[5]:10.2f}"
            ),
            justify="left",
            anchor="center"
        )

        progress.stop()
        calculate_button.config(state=tk.NORMAL)  # Re-enable the button after calculation
        is_calculating = False   # Reset the calculation status
    except queue.Empty:
        root.after(200, update_gui)  # Check every 200 milliseconds

n = 130  # Number of Bayesian iterations

def calculate():
    global is_calculating
    if is_calculating:  # If calculation is already in progress, return immediately
        return

    is_calculating = True  # Set as calculating
    calculate_button.config(state=tk.DISABLED)  # Disable the button

     # Check if any input field is empty
    if not (rx_entry.get() and e_entry.get() and rg_entry.get() and rb_entry.get()):
        result_label.config(text="\n\n        Please enter a valid numerical value！", fg="red")
        calculate_button.config(state=tk.NORMAL)   # Re-enable the button
        is_calculating = False  # Reset status
        return
    
    try:
        # Convert input values to float
        Rx = float(rx_entry.get())
        E = float(e_entry.get())
        rg = float(rg_entry.get())
        Rb = float(rb_entry.get())
    except ValueError:
        result_label.config(text="\n\n        Please enter a valid numerical value！", fg="red")
        calculate_button.config(state=tk.NORMAL) 
        is_calculating = False  
        return

    # Reset progress bar
    progress['value'] = 0
    progress['maximum'] = n  # Set the progress bar maximum value to the number of iterations
    
    def update_progress(result):
        progress['value'] += 1
        root.update_idletasks()  # Update the interface

    def run_calculation():
        best_values = search_around_best_values(Rx, Rb, rg, E, update_progress, n)
        progress_queue.put(best_values)   # Put the result into the queue

     # Run calculation in a separate thread
    threading.Thread(target=run_calculation).start()
    update_gui()  # Start the cycle of updating the interface

# Create Calculation Button
calculate_button = tk.Button(root, text="Optimize!", command=calculate, bg="blue", fg="white", font=("Times New Roman", 16))
calculate_button.grid(row=10, columnspan=3, padx=3, pady=20)

# Load and display image
if getattr(sys, 'frozen', False):
    img_path = os.path.join(sys._MEIPASS, "Wheatstone Bridge.jpg")
else:
    img_path = os.path.join(os.getcwd(), "Wheatstone Bridge.jpg")

img = Image.open(img_path).resize((220, 180), Image.LANCZOS)
img_tk = ImageTk.PhotoImage(img)

image_label = tk.Label(root, image=img_tk)
image_label.grid(row=0, column=2, rowspan=4, padx=5, pady=5, sticky='nsew')
image_label.image = img_tk  

root.mainloop()