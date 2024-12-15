import pickle
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from seaborn import color_palette


class ModelPerformanceTracker:
    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(list))
        self.metrics = {}
        self.test_results = {}  # 테스트 결과 저장

    def update(self, model_name, epoch, train_loss, valid_loss, epoch_time, params_count):
        self.history[model_name]['train_loss'].append(train_loss)
        self.history[model_name]['valid_loss'].append(valid_loss)
        self.history[model_name]['epoch_time'].append(epoch_time)
        self.metrics[model_name] = {
            'params_count': params_count,
            'total_time': sum(self.history[model_name]['epoch_time']),
            'best_valid_loss': min(self.history[model_name]['valid_loss']),
            'convergence_epoch': np.argmin(self.history[model_name]['valid_loss']) + 1
        }

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def delete_model(self, model_name):
        if model_name in self.history:
            del self.history[model_name]
            del self.metrics[model_name]
            if model_name in self.test_results:
                del self.test_results[model_name]

    def save_test_results(self, model_name, predictions, actuals, cords):
        self.test_results[model_name] = {
            'predictions': predictions,
            'actuals': actuals,
            'cords': cords
        }

# Example usage:
# tracker = ModelPerformanceTracker()
# tracker.update('model1', epoch=1, train_loss=0.5, valid_loss=0.4, epoch_time=60, params_count=1000)
# tracker.save_to_file('tracker.pkl')
# loaded_tracker = ModelPerformanceTracker.load_from_file('tracker.pkl')
# loaded_tracker.delete_model('model1')

    def plot_test_result(self, model_name):
        # Convert cords to a numpy array if it is a list of numpy arrays
        cords = np.array(self.test_results[model_name]['cords'])  # (x, y) coordinates
        actuals = np.array(self.test_results[model_name]['actuals'])  # Corresponding actual values for z-axis
        predictions = np.array(self.test_results[model_name]['predictions'])  # Corresponding predicted values for z-axis

        # 스타일 적용
        plt.style.use("seaborn-v0_8-poster")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (18, 10)
        })

        # Create a 3D plot
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for actuals and predictions in 3D
        scatter_actuals = ax.scatter(cords[:, :, 0], cords[:, :, 1], actuals, c=actuals, cmap='Blues', label='Actual Values', marker='o', s=100, alpha=0.7)
        scatter_predictions = ax.scatter(cords[:, :, 0], cords[:, :, 1], predictions, c=predictions, cmap='Oranges', label='Predicted Values', marker='x', s=100, alpha=0.7)

        ax.set_title(f'Test Results for {model_name}', pad=15)
        ax.set_xlabel('X Coordinate', labelpad=10)
        ax.set_ylabel('Y Coordinate', labelpad=10)
        ax.set_zlabel('Values (Actuals/Predictions)', labelpad=10)
        ax.legend(loc='best', frameon=True)
        ax.grid(visible=True, linestyle='--', alpha=0.6)

        # Add color bars for better interpretation of the color mapping
        fig.colorbar(scatter_actuals, ax=ax, label='Actual Values')
        fig.colorbar(scatter_predictions, ax=ax, label='Predicted Values')

        plt.tight_layout(pad=3.0)
        plt.show()


    def plot_performance_comparison(self):
        plt.style.use("seaborn-v0_8-poster")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (15, 10)
        })

        fig = plt.figure(figsize=(18, 18))

        # 1. Learning Curves
        ax1 = fig.add_subplot(221)
        for model_name in self.history:
            ax1.plot(self.history[model_name]['train_loss'],
                     label=f'{model_name} Train', linewidth=2)
            ax1.plot(self.history[model_name]['valid_loss'],
                     '--', label=f'{model_name} Valid', linewidth=2)
        ax1.set_title('Learning Curves', pad=10)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right', frameon=True)
        ax1.grid(visible=True, linestyle='--', alpha=0.6)

        # 2. Training Time per Epoch
        ax2 = fig.add_subplot(222)
        for model_name in self.history:
            ax2.plot(self.history[model_name]['epoch_time'],
                     label=model_name, linewidth=2)
        ax2.set_title('Training Time per Epoch', pad=10)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (seconds)')
        ax2.legend(loc='upper right', frameon=True)
        ax2.grid(visible=True, linestyle='--', alpha=0.6)

        # 3. Parameter Count Comparison
        ax3 = fig.add_subplot(223)
        params_counts = [self.metrics[m]['params_count'] for m in self.metrics]
        bar_colors = color_palette("pastel", len(self.metrics))
        ax3.bar(self.metrics.keys(), params_counts, color=bar_colors, edgecolor='black')
        ax3.set_title('Parameter Count Comparison', pad=10)
        ax3.set_ylabel('Number of Parameters')
        ax3.set_xlabel('Model')
        for i, v in enumerate(params_counts):
            ax3.text(i, v + 0.05 * max(params_counts), f'{v}', ha='center', fontsize=10)

        # 4. Performance Summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        summary_text = "Performance Summary:\n\n"
        for model_name, metrics in self.metrics.items():
            summary_text += f"**{model_name}**:\n"
            summary_text += f"• Parameters: {metrics['params_count']:,}\n"
            summary_text += f"• Best Valid Loss: {metrics['best_valid_loss']:.4f}\n"
            summary_text += f"• Total Training Time: {metrics['total_time']:.2f}s\n"
            summary_text += f"• Convergence Epoch: {metrics['convergence_epoch']}\n\n"
        ax4.text(0, 1, summary_text, va='top', fontsize=12, wrap=True)

        # Adjust layout and add title
        plt.tight_layout(pad=3.0)
        plt.suptitle("Model Performance Comparison", fontsize=20, y=1.02)

        plt.show()
