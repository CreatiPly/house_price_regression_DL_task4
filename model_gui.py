import tkinter as tk


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
        tk.Button(control_frame, text="Load Dataset", command=self.dummy_action).pack(
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

    def dummy_action(self):
        print("Button clicked - Feature coming soon!")


if __name__ == "__main__":
    root = tk.Tk()
    app = HousingPriceInferenceGUI(root)
    root.mainloop()
