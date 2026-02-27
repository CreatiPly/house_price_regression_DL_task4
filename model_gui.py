import tkinter as tk
from tkinter import messagebox, filedialog
import torch
import torch.nn as nn
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class SimpleRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearLayer1 = nn.Linear(8, 50)
        self.linearLayer2 = nn.Linear(50, 100)
        self.relu = nn.ReLU()
        self.linearLayer3 = nn.Linear(100, 1)

    def forward(self, x):
        u = self.linearLayer1(x)
        v = self.relu(u)
        w = self.linearLayer2(v)
        m = self.relu(w)
        return self.linearLayer3(m)


class HousingPriceInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Housing Price Regression Inference")
        self.root.geometry("1200x800")

        self.model = None
        self.scaler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            pass

        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(
            control_frame,
            text="Load Model",
            bg="blue",
            fg="white",
            command=self.load_model,
        ).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Load Dataset", command=self.load_dataset).pack(
            fill=tk.X, pady=2
        )

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        self.entries = {}
        features = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        for i, feature in enumerate(features):
            row, col = i // 4, (i % 4) * 2
            tk.Label(input_frame, text=feature).grid(
                row=row, column=col, sticky=tk.E, padx=5, pady=2
            )
            entry = tk.Entry(input_frame, width=15)
            entry.grid(row=row, column=col + 1, padx=5, pady=2)
            self.entries[feature] = entry

        tk.Button(
            self.root,
            text="Run Inference Manual Data",
            command=self.run_manual_inference,
        ).pack(pady=5)

        list_frame = tk.Frame(self.root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        train_frame = tk.Frame(list_frame)
        train_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(train_frame, text="Train Data:").pack(anchor=tk.W)
        self.train_listbox = tk.Listbox(train_frame, width=50)
        self.train_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.train_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        test_frame = tk.Frame(list_frame)
        test_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(test_frame, text="Test Data:").pack(anchor=tk.W)
        self.test_listbox = tk.Listbox(test_frame, width=50)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # --- Bound click event to inference logic ---
        self.test_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        output_frame = tk.Frame(self.root)
        output_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        self.true_price_label = tk.Label(
            output_frame,
            text="True Price: N/A",
            bg="lightgreen",
            font=("Arial", 12),
            width=40,
            anchor=tk.W,
        )
        self.true_price_label.pack(side=tk.LEFT, padx=5)
        self.pred_price_label = tk.Label(
            output_frame,
            text="Predicted Price: N/A",
            bg="lightcoral",
            font=("Arial", 12),
            width=40,
            anchor=tk.W,
        )
        self.pred_price_label.pack(side=tk.RIGHT, padx=5)

    def load_model(self):
        filepath = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if filepath:
            try:
                checkpoint = torch.load(
                    filepath, map_location=self.device, weights_only=False
                )
                self.model = SimpleRegressionNet().to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")

    def load_dataset(self):
        self.train_listbox.delete(0, tk.END)
        self.test_listbox.delete(0, tk.END)
        housing = fetch_california_housing()
        X_train, X_val, y_train, y_val = train_test_split(
            housing.data, housing.target, test_size=0.2, random_state=42
        )
        for i in range(min(100, len(X_train))):
            features_str = ", ".join([f"{val:.4f}" for val in X_train[i]])
            item = f"{features_str}, True Label: {y_train[i]:.4f}"
            self.train_listbox.insert(tk.END, item)
        for i in range(min(100, len(X_val))):
            features_str = ", ".join([f"{val:.4f}" for val in X_val[i]])
            item = f"{features_str}, True Label: {y_val[i]:.4f}"
            self.test_listbox.insert(tk.END, item)

    def on_listbox_select(self, event):
        if not self.model or not self.scaler:
            return
        widget = event.widget
        selection = widget.curselection()
        if not selection:
            return
        item_text = widget.get(selection[0])
        try:
            parts = item_text.split(",")
            features = [float(p.strip()) for p in parts[:8]]
            true_price = parts[-1].split(":")[-1].strip()

            scaled_features = self.scaler.transform([features])
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(
                self.device
            )

            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            self.true_price_label.config(text=f"True Price: ${true_price}")
            self.pred_price_label.config(text=f"Predicted Price: ${prediction:.3f}")
        except Exception as e:
            messagebox.showerror("Parsing Error", str(e))

    def run_manual_inference(self):
        if not self.model or not self.scaler:
            messagebox.showwarning(
                "Warning", "Please load the model and ensure scaler.pkl is present."
            )
            return
        try:
            features = [float(self.entries[f].get()) for f in self.entries]
            scaled_features = self.scaler.transform([features])
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(
                self.device
            )

            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            self.true_price_label.config(text="True Price: Manual Input")
            self.pred_price_label.config(text=f"Predicted Price: ${prediction:.3f}")
        except ValueError:
            messagebox.showerror(
                "Error", "Check your 8 input numbers. Ensure they are valid floats."
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = HousingPriceInferenceGUI(root)
    root.mainloop()
