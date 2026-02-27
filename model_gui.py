import tkinter as tk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class HousingPriceInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Housing Price Regression Inference")
        self.root.geometry("1200x800")
        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(
            control_frame,
            text="Load Model",
            bg="blue",
            fg="white",
            command=self.dummy_action,
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
            self.root, text="Run Inference Manual Data", command=self.dummy_action
        ).pack(pady=5)

        list_frame = tk.Frame(self.root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        train_frame = tk.Frame(list_frame)
        train_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(train_frame, text="Train Data:").pack(anchor=tk.W)
        self.train_listbox = tk.Listbox(train_frame, width=50)
        self.train_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        test_frame = tk.Frame(list_frame)
        test_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(test_frame, text="Test Data:").pack(anchor=tk.W)
        self.test_listbox = tk.Listbox(test_frame, width=50)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

    def load_dataset(self):
        self.train_listbox.delete(0, tk.END)
        self.test_listbox.delete(0, tk.END)
        housing = fetch_california_housing()
        X_train, X_val, y_train, y_val = train_test_split(
            housing.data, housing.target, test_size=0.2, random_state=42
        )

        for i in range(min(100, len(X_train))):
            features_str = ", ".join([f"{val:.4f}" for val in X_train[i]])
            self.train_listbox.insert(
                tk.END, f"{features_str}, True Label: {y_train[i]:.4f}"
            )

        for i in range(min(100, len(X_val))):
            features_str = ", ".join([f"{val:.4f}" for val in X_val[i]])
            self.test_listbox.insert(
                tk.END, f"{features_str}, True Label: {y_val[i]:.4f}"
            )

    def dummy_action(self, *args):
        print("Action triggered - feature coming soon!")


if __name__ == "__main__":
    root = tk.Tk()
    app = HousingPriceInferenceGUI(root)
    root.mainloop()
