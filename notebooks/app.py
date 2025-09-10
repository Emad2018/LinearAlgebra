import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import os
import tempfile
import subprocess
import sys
from pathlib import Path
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class LinearAlgebraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Algebra Visualizer")
        self.root.geometry("1200x800")
        
        # Setup main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        
        # Vector operations tab
        self.vector_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.vector_tab, text="Vector Operations")
        
        # Matrix transformations tab
        self.matrix_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.matrix_tab, text="Matrix Transformations")
        
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Setup the UI components
        self.setup_vector_operations()
        self.setup_matrix_transformations()
        self.setup_animation_viewer()
        
        # Initialize data structures
        self.vectors = []
        self.matrices = []
        self.animation_path = None
        
    def setup_vector_operations(self):
        # Left panel for vector input
        left_panel = ttk.LabelFrame(self.vector_tab, text="Vector Input", padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Vector dimensions
        ttk.Label(left_panel, text="Vector Dimension:").grid(row=0, column=0, sticky="w")
        self.dim_var = tk.StringVar(value="2")
        dim_combo = ttk.Combobox(left_panel, textvariable=self.dim_var, values=["2", "3"], width=5)
        dim_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        dim_combo.bind("<<ComboboxSelected>>", self.update_vector_input)
        
        # Frame for vector inputs
        self.vector_input_frame = ttk.Frame(left_panel)
        self.vector_input_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        # Button frame
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        add_btn = ttk.Button(button_frame, text="Add Vector", command=self.add_vector)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        random_btn = ttk.Button(button_frame, text="Random Vector", command=self.add_random_vector)
        random_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear Vectors", command=self.clear_vectors)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel for vector list and operations
        right_panel = ttk.LabelFrame(self.vector_tab, text="Vector Operations", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Vector list
        ttk.Label(right_panel, text="Current Vectors:").grid(row=0, column=0, sticky="w")
        self.vector_listbox = tk.Listbox(right_panel, height=10, width=30)
        self.vector_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Operations frame
        op_frame = ttk.LabelFrame(right_panel, text="Operations")
        op_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Operation selection
        ttk.Label(op_frame, text="Select Operation:").grid(row=0, column=0, sticky="w")
        self.op_var = tk.StringVar(value="Addition")
        op_combo = ttk.Combobox(op_frame, textvariable=self.op_var, 
                               values=["Addition", "Subtraction", "Dot Product", "Cross Product", "Scalar Multiplication"])
        op_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # For scalar multiplication
        ttk.Label(op_frame, text="Scalar Value:").grid(row=1, column=0, sticky="w")
        self.scalar_var = tk.StringVar(value="2")
        scalar_entry = ttk.Entry(op_frame, textvariable=self.scalar_var, width=10)
        scalar_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Calculate button
        calc_btn = ttk.Button(op_frame, text="Calculate & Animate", command=self.perform_vector_operation)
        calc_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Initialize vector input
        self.update_vector_input()
        
        # Configure grid weights
        self.vector_tab.columnconfigure(0, weight=1)
        self.vector_tab.columnconfigure(1, weight=1)
        self.vector_tab.rowconfigure(0, weight=1)
        
    def setup_matrix_transformations(self):
        # Left panel for matrix input
        left_panel = ttk.LabelFrame(self.matrix_tab, text="Matrix Input", padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Matrix dimensions
        ttk.Label(left_panel, text="Matrix Size:").grid(row=0, column=0, sticky="w")
        self.matrix_dim_var = tk.StringVar(value="2x2")
        matrix_dim_combo = ttk.Combobox(left_panel, textvariable=self.matrix_dim_var, 
                                        values=["2x2", "3x3"], width=5)
        matrix_dim_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        matrix_dim_combo.bind("<<ComboboxSelected>>", self.update_matrix_input)
        
        # Frame for matrix inputs
        self.matrix_input_frame = ttk.Frame(left_panel)
        self.matrix_input_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        # Button frame
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        add_btn = ttk.Button(button_frame, text="Add Matrix", command=self.add_matrix)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Common transformation buttons
        common_frame = ttk.LabelFrame(left_panel, text="Common Transformations")
        common_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        rotation_btn = ttk.Button(common_frame, text="Rotation", command=lambda: self.add_common_matrix("rotation"))
        rotation_btn.grid(row=0, column=0, padx=5, pady=5)
        
        scaling_btn = ttk.Button(common_frame, text="Scaling", command=lambda: self.add_common_matrix("scaling"))
        scaling_btn.grid(row=0, column=1, padx=5, pady=5)
        
        shear_btn = ttk.Button(common_frame, text="Shear", command=lambda: self.add_common_matrix("shear"))
        shear_btn.grid(row=0, column=2, padx=5, pady=5)
        
        reflection_btn = ttk.Button(common_frame, text="Reflection", command=lambda: self.add_common_matrix("reflection"))
        reflection_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Right panel for matrix operations
        right_panel = ttk.LabelFrame(self.matrix_tab, text="Matrix Operations", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Matrix list
        ttk.Label(right_panel, text="Current Matrices:").grid(row=0, column=0, sticky="w")
        self.matrix_listbox = tk.Listbox(right_panel, height=10, width=30)
        self.matrix_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Operations frame
        op_frame = ttk.LabelFrame(right_panel, text="Operations")
        op_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Operation selection
        ttk.Label(op_frame, text="Select Operation:").grid(row=0, column=0, sticky="w")
        self.matrix_op_var = tk.StringVar(value="Apply to Vector")
        matrix_op_combo = ttk.Combobox(op_frame, textvariable=self.matrix_op_var, 
                                       values=["Apply to Vector", "Matrix Multiplication", "Determinant", "Inverse"])
        matrix_op_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Calculate button
        calc_btn = ttk.Button(op_frame, text="Calculate & Animate", command=self.perform_matrix_operation)
        calc_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Initialize matrix input
        self.update_matrix_input()
        
    def setup_animation_viewer(self):
        # Animation viewer frame
        self.animation_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="10")
        self.animation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup matplotlib figure for visualization
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.animation_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation controls
        control_frame = ttk.Frame(self.animation_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        save_btn = ttk.Button(control_frame, text="Save Animation", command=self.save_animation)
        save_btn.pack(side=tk.LEFT, padx=5)
        
    def update_vector_input(self, event=None):
        # Clear previous input widgets
        for widget in self.vector_input_frame.winfo_children():
            widget.destroy()
            
        dim = int(self.dim_var.get())
        self.vector_entries = []
        
        # Create input fields based on dimension
        for i in range(dim):
            ttk.Label(self.vector_input_frame, text=f"Component {i+1}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            entry = ttk.Entry(self.vector_input_frame, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            entry.insert(0, "0")
            self.vector_entries.append(entry)
    
    def update_matrix_input(self, event=None):
        # Clear previous input widgets
        for widget in self.matrix_input_frame.winfo_children():
            widget.destroy()
            
        dim_str = self.matrix_dim_var.get()
        rows, cols = map(int, dim_str.split("x"))
        
        self.matrix_entries = []
        
        # Create input fields for matrix
        for i in range(rows):
            row_entries = []
            for j in range(cols):
                entry = ttk.Entry(self.matrix_input_frame, width=5)
                entry.grid(row=i, column=j, padx=3, pady=3)
                entry.insert(0, "1" if i == j else "0")  # Identity matrix by default
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
    
    def add_vector(self):
        try:
            dim = int(self.dim_var.get())
            vector = np.zeros(dim)
            
            for i in range(dim):
                vector[i] = float(self.vector_entries[i].get())
                
            self.vectors.append(vector)
            self.update_vector_list()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for vector components")
    
    def add_random_vector(self):
        dim = int(self.dim_var.get())
        vector = np.random.uniform(-10, 10, dim)
        self.vectors.append(vector)
        self.update_vector_list()
    
    def clear_vectors(self):
        self.vectors = []
        self.update_vector_list()
    
    def update_vector_list(self):
        self.vector_listbox.delete(0, tk.END)
        for i, v in enumerate(self.vectors):
            self.vector_listbox.insert(tk.END, f"Vector {i+1}: {v}")
    
    def add_matrix(self):
        try:
            dim_str = self.matrix_dim_var.get()
            rows, cols = map(int, dim_str.split("x"))
            
            matrix = np.zeros((rows, cols))
            
            for i in range(rows):
                for j in range(cols):
                    matrix[i, j] = float(self.matrix_entries[i][j].get())
            
            self.matrices.append(matrix)
            self.update_matrix_list()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for matrix elements")
    
    def add_common_matrix(self, type_):
        dim_str = self.matrix_dim_var.get()
        dim = int(dim_str[0])  # Assuming square matrices
        
        if dim == 2:
            if type_ == "rotation":
                angle = float(simpledialog.askstring("Angle", "Enter rotation angle in degrees:") or "0")
                angle_rad = np.radians(angle)
                matrix = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]
                ])
            elif type_ == "scaling":
                sx = float(simpledialog.askstring("Scale X", "Enter x scale factor:") or "1")
                sy = float(simpledialog.askstring("Scale Y", "Enter y scale factor:") or "1")
                matrix = np.array([[sx, 0], [0, sy]])
            elif type_ == "shear":
                shx = float(simpledialog.askstring("Shear X", "Enter x shear factor:") or "0")
                shy = float(simpledialog.askstring("Shear Y", "Enter y shear factor:") or "0")
                matrix = np.array([[1, shx], [shy, 1]])
            elif type_ == "reflection":
                axis = simpledialog.askstring("Reflection", "Enter axis (x, y, or xy for origin):") or "x"
                if axis.lower() == "x":
                    matrix = np.array([[1, 0], [0, -1]])
                elif axis.lower() == "y":
                    matrix = np.array([[-1, 0], [0, 1]])
                else:  # origin
                    matrix = np.array([[-1, 0], [0, -1]])
        else:  # 3D
            messagebox.showinfo("Not Implemented", "3D transformations are not fully implemented yet")
            return
            
        self.matrices.append(matrix)
        self.update_matrix_list()
    
    def update_matrix_list(self):
        self.matrix_listbox.delete(0, tk.END)
        for i, m in enumerate(self.matrices):
            self.matrix_listbox.insert(tk.END, f"Matrix {i+1}: {m.shape}")
    
    def perform_vector_operation(self):
        if len(self.vectors) < 1:
            messagebox.showerror("Error", "Need at least one vector for operations")
            return
            
        operation = self.op_var.get()
        
        if operation in ["Addition", "Subtraction"] and len(self.vectors) < 2:
            messagebox.showerror("Error", "Need at least two vectors for this operation")
            return
            
        if operation == "Cross Product" and (len(self.vectors) < 2 or self.vectors[0].size != 3 or self.vectors[1].size != 3):
            messagebox.showerror("Error", "Cross product requires two 3D vectors")
            return
            
        # Visualize the operation
        self.visualize_vector_operation(operation)
        
        # Generate Manim code in a separate thread
        threading.Thread(target=self.generate_vector_animation, args=(operation,)).start()
        
    def perform_matrix_operation(self):
        if len(self.matrices) < 1:
            messagebox.showerror("Error", "Need at least one matrix for operations")
            return
            
        operation = self.matrix_op_var.get()
        
        if operation == "Apply to Vector" and len(self.vectors) < 1:
            messagebox.showerror("Error", "Need at least one vector to apply matrix to")
            return
            
        if operation == "Matrix Multiplication" and len(self.matrices) < 2:
            messagebox.showerror("Error", "Need at least two matrices for multiplication")
            return
            
        # Visualize the operation
        self.visualize_matrix_operation(operation)
        
        # Generate Manim code in a separate thread
        threading.Thread(target=self.generate_matrix_animation, args=(operation,)).start()
        
    def visualize_vector_operation(self, operation):
        # Clear previous plot
        self.fig.clear()
        
        # Determine if we need 2D or 3D plot
        dim = self.vectors[0].size
        
        if dim == 2:
            ax = self.fig.add_subplot(111)
            
            # Plot coordinate system
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Set limits based on vectors
            max_val = max([np.max(np.abs(v)) for v in self.vectors]) * 1.2
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            
            # Plot vectors
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            
            for i, v in enumerate(self.vectors):
                ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                         color=colors[i % len(colors)], label=f"v{i+1}")
                
            # Calculate and plot result based on operation
            if operation == "Addition":
                result = np.sum(self.vectors, axis=0)
                ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, 
                         color='k', label="Result")
                ax.set_title("Vector Addition")
                
            elif operation == "Subtraction":
                result = self.vectors[0] - self.vectors[1]
                ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, 
                         color='k', label="Result")
                ax.set_title("Vector Subtraction")
                
            elif operation == "Dot Product":
                dot_product = np.dot(self.vectors[0], self.vectors[1])
                ax.set_title(f"Dot Product: {dot_product:.2f}")
                
            elif operation == "Scalar Multiplication":
                scalar = float(self.scalar_var.get())
                result = scalar * self.vectors[0]
                ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, 
                         color='k', label=f"{scalar}*v1")
                ax.set_title(f"Scalar Multiplication: {scalar} * v1")
                
            ax.legend()
            
        elif dim == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            
            # Set limits based on vectors
            max_val = max([np.max(np.abs(v)) for v in self.vectors]) * 1.2
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
            
            # Plot vectors
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            
            for i, v in enumerate(self.vectors):
                ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[i % len(colors)], label=f"v{i+1}")
                
            # Calculate and plot result based on operation
            if operation == "Addition":
                result = np.sum(self.vectors, axis=0)
                ax.quiver(0, 0, 0, result[0], result[1], result[2], color='k', label="Result")
                ax.set_title("Vector Addition")
                
            elif operation == "Subtraction":
                result = self.vectors[0] - self.vectors[1]
                ax.quiver(0, 0, 0, result[0], result[1], result[2], color='k', label="Result")
                ax.set_title("Vector Subtraction")
                
            elif operation == "Dot Product":
                dot_product = np.dot(self.vectors[0], self.vectors[1])
                ax.set_title(f"Dot Product: {dot_product:.2f}")
                
            elif operation == "Cross Product":
                result = np.cross(self.vectors[0], self.vectors[1])
                ax.quiver(0, 0, 0, result[0], result[1], result[2], color='k', label="Result")
                ax.set_title("Cross Product")
                
            elif operation == "Scalar Multiplication":
                scalar = float(self.scalar_var.get())
                result = scalar * self.vectors[0]
                ax.quiver(0, 0, 0, result[0], result[1], result[2], color='k', label=f"{scalar}*v1")
                ax.set_title(f"Scalar Multiplication: {scalar} * v1")
                
            ax.legend()
            
        self.canvas.draw()
        
    def visualize_matrix_operation(self, operation):
        # Clear previous plot
        self.fig.clear()
        
        if operation == "Apply to Vector":
            # Get the first matrix and vector
            matrix = self.matrices[0]
            vector = self.vectors[0]
            
            # Check dimensions
            if matrix.shape[1] != vector.size:
                messagebox.showerror("Error", f"Matrix columns ({matrix.shape[1]}) must match vector dimension ({vector.size})")
                return
                
            # Calculate the transformed vector
            result = np.dot(matrix, vector)
            
            # Visualize the transformation
            if vector.size == 2:
                ax = self.fig.add_subplot(111)
                
                # Plot coordinate system
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                
                # Set limits
                max_val = max(np.max(np.abs(vector)), np.max(np.abs(result))) * 1.2
                ax.set_xlim([-max_val, max_val])
                ax.set_ylim([-max_val, max_val])
                
                # Plot original vector
                ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, 
                         color='b', label="Original")
                
                # Plot transformed vector
                ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, 
                         color='r', label="Transformed")
                
                # Add grid lines to show the transformation
                # For clarity, we'll add basis vectors transformation
                e1 = np.array([1, 0])
                e2 = np.array([0, 1])
                
                e1_transformed = np.dot(matrix, e1)
                e2_transformed = np.dot(matrix, e2)
                
                ax.quiver(0, 0, e1_transformed[0], e1_transformed[1], angles='xy', scale_units='xy', scale=1, 
                         color='g', alpha=0.5, label="e1 transformed")
                ax.quiver(0, 0, e2_transformed[0], e2_transformed[1], angles='xy', scale_units='xy', scale=1, 
                         color='m', alpha=0.5, label="e2 transformed")
                
                ax.legend()
                ax.set_title("Matrix Transformation")
                
            elif vector.size == 3:
                ax = self.fig.add_subplot(111, projection='3d')
                
                # Set limits
                max_val = max(np.max(np.abs(vector)), np.max(np.abs(result))) * 1.2
                ax.set_xlim([-max_val, max_val])
                ax.set_ylim([-max_val, max_val])
                ax.set_zlim([-max_val, max_val])
                
                # Plot original vector
                ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='b', label="Original")
                
                # Plot transformed vector
                ax.quiver(0, 0, 0, result[0], result[1], result[2], color='r', label="Transformed")
                
                ax.legend()
                ax.set_title("Matrix Transformation")
        
        elif operation == "Matrix Multiplication":
            # Get the matrices
            matrix1 = self.matrices[0]
            matrix2 = self.matrices[1]
            
            # Check dimensions
            if matrix1.shape[1] != matrix2.shape[0]:
                messagebox.showerror("Error", f"Matrix 1 columns ({matrix1.shape[1]}) must match Matrix 2 rows ({matrix2.shape[0]})")
                return
                
            # Calculate the result
            result = np.dot(matrix1, matrix2)
            
            # Plot as text
            ax = self.fig.add_subplot(111)
            ax.axis('off')
            
            text = f"Matrix 1:\n{matrix1}\n\nMatrix 2:\n{matrix2}\n\nResult:\n{result}"
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
            
        elif operation == "Determinant":
            # Get the matrix
            matrix = self.matrices[0]
            
            # Check if square
            if matrix.shape[0] != matrix.shape[1]:
                messagebox.showerror("Error", "Determinant requires a square matrix")
                return
                
            # Calculate determinant
            det = np.linalg.det(matrix)
            
            # Plot as text
            ax = self.fig.add_subplot(111)
            ax.axis('off')
            
            text = f"Matrix:\n{matrix}\n\nDeterminant: {det:.4f}"
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
            
        elif operation == "Inverse":
            # Get the matrix
            matrix = self.matrices[0]
            
            # Check if square
            if matrix.shape[0] != matrix.shape[1]:
                messagebox.showerror("Error", "Inverse requires a square matrix")
                return
                
            # Check if invertible
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                messagebox.showerror("Error", "Matrix is not invertible (determinant ≈ 0)")
                return
                
            # Calculate inverse
            inverse = np.linalg.inv(matrix)
            
            # Plot as text
            ax = self.fig.add_subplot(111)
            ax.axis('off')
            
            text = f"Matrix:\n{matrix}\n\nInverse:\n{inverse}"
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
            
        self.canvas.draw()
    
    def generate_vector_animation(self, operation):
        # This would generate a Manim script and run it
        # For simplicity, we'll just show a message
        messagebox.showinfo("Manim Integration", 
                           "Manim integration would generate an animation here.\n\n"
                           "To fully implement this, you would need to:\n"
                           "1. Generate a Manim script based on the operation\n"
                           "2. Run the script with subprocess\n"
                           "3. Display the resulting video")
    
    def generate_matrix_animation(self, operation):
        # Similar to generate_vector_animation
        messagebox.showinfo("Manim Integration", 
                           "Manim integration would generate a matrix transformation animation here.")
    
    def save_animation(self):
        # This would save the matplotlib figure
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            self.fig.savefig(file_path)
            messagebox.showinfo("Save", f"Visualization saved to {file_path}")

def main():
    root = tk.Tk()
    app = LinearAlgebraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()