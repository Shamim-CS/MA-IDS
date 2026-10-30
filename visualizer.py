"""
Figures for the paper: confusion matrices, per-class metrics, distributions,
learning curves and feature analysis.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from logger import IDSLogger


class ResearchVisualizer:
    """Generate publication-quality figures for research papers"""
    
    def __init__(
        self,
        output_dir: str,
        valid_labels: List[str],
        logger: IDSLogger,
        timestamp: str,
        dpi: int = 300,
        style: str = 'seaborn-v0_8-paper'
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.valid_labels = valid_labels
        self.logger = logger
        self.timestamp = timestamp
        self.dpi = dpi
        
        # Set publication-quality style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        
        sns.set_palette("husl")
        
        # Create subdirectories for different figure types
        self.fig_dirs = {
            'confusion': self.output_dir / 'confusion_matrices',
            'performance': self.output_dir / 'performance_metrics',
            'distribution': self.output_dir / 'data_distributions',
            'temporal': self.output_dir / 'temporal_analysis',
            'comparison': self.output_dir / 'model_comparison',
            'learning': self.output_dir / 'learning_curves'
        }
        
        for dir_path in self.fig_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_figure(self, fig, filename: str, subdir: str = 'performance'):
        """Save figure with consistent settings"""
        filepath = self.fig_dirs[subdir] / f"{filename}_{self.timestamp}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.logger.info(f"   Saved: {filepath}")
        return filepath
    
    # 1. CONFUSION MATRIX VARIANTS
    
    def plot_confusion_matrix_standard(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Path:
        """Standard confusion matrix with counts"""
        cm = confusion_matrix(y_true, y_pred, labels=self.valid_labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.valid_labels,
            yticklabels=self.valid_labels,
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        ax.set_title('Confusion Matrix - Classification Results', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return self.save_figure(fig, 'confusion_matrix_standard', 'confusion')
    
    def plot_confusion_matrix_normalized(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Path:
        """Normalized confusion matrix (percentages)"""
        cm = confusion_matrix(y_true, y_pred, labels=self.valid_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            xticklabels=self.valid_labels,
            yticklabels=self.valid_labels,
            cbar_kws={'label': 'Percentage'},
            vmin=0,
            vmax=1,
            ax=ax
        )
        
        ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return self.save_figure(fig, 'confusion_matrix_normalized', 'confusion')
    
    def plot_confusion_matrix_combined(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Path:
        """Side-by-side confusion matrices (count and normalized)"""
        cm = confusion_matrix(y_true, y_pred, labels=self.valid_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Count matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.valid_labels,
            yticklabels=self.valid_labels,
            cbar_kws={'label': 'Count'},
            ax=ax1
        )
        ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=11)
        ax1.set_xlabel('Predicted Label', fontsize=11)
        
        # Normalized matrix
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            xticklabels=self.valid_labels,
            yticklabels=self.valid_labels,
            cbar_kws={'label': 'Percentage'},
            vmin=0,
            vmax=1,
            ax=ax2
        )
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.set_xlabel('Predicted Label', fontsize=11)
        
        plt.tight_layout()
        return self.save_figure(fig, 'confusion_matrix_combined', 'confusion')
    
    # 2. PERFORMANCE METRICS
    
    def plot_per_class_metrics(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Path:
        """Bar chart of precision, recall, F1-score per class"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.valid_labels, zero_division=0
        )
        
        x = np.arange(len(self.valid_labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Attack Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.valid_labels, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return self.save_figure(fig, 'per_class_metrics', 'performance')
    
    def plot_metrics_radar(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Path:
        """Radar chart of metrics per class"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.valid_labels, zero_division=0
        )
        
        # Number of variables
        num_vars = len(self.valid_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the circle
        precision = np.concatenate((precision, [precision[0]]))
        recall = np.concatenate((recall, [recall[0]]))
        f1 = np.concatenate((f1, [f1[0]]))
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, precision, 'o-', linewidth=2, label='Precision', color='#1f77b4')
        ax.fill(angles, precision, alpha=0.25, color='#1f77b4')
        
        ax.plot(angles, recall, 'o-', linewidth=2, label='Recall', color='#ff7f0e')
        ax.fill(angles, recall, alpha=0.25, color='#ff7f0e')
        
        ax.plot(angles, f1, 'o-', linewidth=2, label='F1-Score', color='#2ca02c')
        ax.fill(angles, f1, alpha=0.25, color='#2ca02c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.valid_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        ax.set_title('Performance Metrics Radar Chart', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return self.save_figure(fig, 'metrics_radar', 'performance')
    
    def plot_accuracy_comparison(
        self,
        model_results: Dict[str, float]
    ) -> Path:
        """Compare accuracy across different models/configurations"""
        models = list(model_results.keys())
        accuracies = list(model_results.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", len(models))
        bars = ax.barh(models, accuracies, color=colors, alpha=0.8)
        
        ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 100])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 1, i, f'{acc:.2f}%', 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, 'accuracy_comparison', 'comparison')
    
    # 3. DATA DISTRIBUTION VISUALIZATIONS
    
    def plot_class_distribution(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Path:
        """Compare distribution of true vs predicted labels"""
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.valid_labels))
        width = 0.35
        
        true_values = [true_counts.get(label, 0) for label in self.valid_labels]
        pred_values = [pred_counts.get(label, 0) for label in self.valid_labels]
        
        bars1 = ax.bar(x - width/2, true_values, width, label='True Distribution', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_values, width, label='Predicted Distribution', alpha=0.8)
        
        ax.set_xlabel('Attack Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution: True vs Predicted', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.valid_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return self.save_figure(fig, 'class_distribution', 'distribution')
    
    def plot_class_balance_pie(self, y_true: List[str]) -> Path:
        """Pie chart showing class imbalance"""
        counts = Counter(y_true)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = [f"{label}\n({counts[label]})" for label in self.valid_labels]
        sizes = [counts[label] for label in self.valid_labels]
        colors = sns.color_palette("husl", len(self.valid_labels))
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        # Bold percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        
        return self.save_figure(fig, 'class_balance_pie', 'distribution')
    
    # 4. LEARNING CURVE & TEMPORAL ANALYSIS
    
    def plot_learning_curve(
        self,
        predictions_over_time: List[Tuple[int, str, str, bool]]
    ) -> Path:
        """
        Plot accuracy improvement over time
        predictions_over_time: List of (index, true_label, pred_label, is_correct)
        """
        indices = [p[0] for p in predictions_over_time]
        correct = [1 if p[3] else 0 for p in predictions_over_time]
        
        # Calculate rolling accuracy
        window_size = max(10, len(correct) // 20)
        rolling_acc = pd.Series(correct).rolling(window=window_size, min_periods=1).mean() * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(indices, rolling_acc, linewidth=2, label=f'Rolling Accuracy (window={window_size})')
        ax.axhline(y=rolling_acc.mean(), color='r', linestyle='--', 
                   label=f'Mean Accuracy: {rolling_acc.mean():.2f}%')
        
        ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Learning Curve: Accuracy Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        return self.save_figure(fig, 'learning_curve', 'learning')
    
    def plot_error_rate_over_time(
        self,
        predictions_over_time: List[Tuple[int, str, str, bool]]
    ) -> Path:
        """Plot error rate progression"""
        indices = [p[0] for p in predictions_over_time]
        errors = [0 if p[3] else 1 for p in predictions_over_time]
        
        # Cumulative error rate
        cumulative_errors = np.cumsum(errors)
        cumulative_total = np.arange(1, len(errors) + 1)
        error_rate = (cumulative_errors / cumulative_total) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(indices, error_rate, linewidth=2, color='red')
        ax.fill_between(indices, error_rate, alpha=0.3, color='red')
        
        ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Error Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Error Rate Progression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        return self.save_figure(fig, 'error_rate_progression', 'learning')
    
    def plot_mistakes_by_class(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Path:
        """Stacked bar showing correct vs incorrect predictions per class"""
        from collections import defaultdict
        
        correct_counts = defaultdict(int)
        incorrect_counts = defaultdict(int)
        
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct_counts[true] += 1
            else:
                incorrect_counts[true] += 1
        
        correct_values = [correct_counts[label] for label in self.valid_labels]
        incorrect_values = [incorrect_counts[label] for label in self.valid_labels]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.valid_labels))
        width = 0.6
        
        bars1 = ax.bar(x, correct_values, width, label='Correct', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, incorrect_values, width, bottom=correct_values, 
                       label='Incorrect', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Attack Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Results by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.valid_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (c, ic) in enumerate(zip(correct_values, incorrect_values)):
            total = c + ic
            if total > 0:
                accuracy = (c / total) * 100
                ax.text(i, total + 5, f'{accuracy:.1f}%', 
                       ha='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        return self.save_figure(fig, 'mistakes_by_class', 'performance')
    
    # 5. ROC AND PRECISION-RECALL CURVES
    
    def plot_roc_curves_multiclass(
        self,
        y_true: List[str],
        y_pred_proba: np.ndarray
    ) -> Path:
        """
        Plot ROC curves for multiclass classification
        y_pred_proba: probability predictions (n_samples, n_classes)
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=self.valid_labels)
        n_classes = len(self.valid_labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("husl", n_classes)
        
        for i, (color, label) in enumerate(zip(colors, self.valid_labels)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{label} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Multiclass Classification', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, 'roc_curves_multiclass', 'performance')
    
    # 6. FEATURE IMPORTANCE & ANALYSIS
    
    def plot_feature_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features_to_plot: List[str]
    ) -> Path:
        """Heatmap of feature correlations"""
        corr_matrix = df[features_to_plot].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return self.save_figure(fig, 'feature_correlation', 'distribution')
    
    # 7. COMPARISON PLOTS
    
    def plot_model_comparison_table(
        self,
        comparison_data: Dict[str, Dict[str, float]]
    ) -> Path:
        """
        Create a comparison table of different models
        comparison_data: {'Model1': {'Accuracy': 0.95, 'Precision': 0.94, ...}, ...}
        """
        df = pd.DataFrame(comparison_data).T
        
        fig, ax = plt.subplots(figsize=(12, len(comparison_data) * 0.8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(df) + 1):
            table[(i, -1)].set_facecolor('#D9E1F2')
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
        
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        
        return self.save_figure(fig, 'model_comparison_table', 'comparison')
    
    # 8. COMPREHENSIVE DASHBOARD
    
    def create_performance_dashboard(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Path:
        """Create a comprehensive dashboard with multiple metrics"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        cm = confusion_matrix(y_true, y_pred, labels=self.valid_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.valid_labels, yticklabels=self.valid_labels,
                   ax=ax1)
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Per-class metrics
        ax2 = fig.add_subplot(gs[0, 2])
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.valid_labels, zero_division=0
        )
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=self.valid_labels)
        
        metrics_df.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Per-Class Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.legend(loc='lower right', fontsize=8)
        ax2.set_ylim([0, 1.05])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Class distribution
        ax3 = fig.add_subplot(gs[1, 2])
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        x = np.arange(len(self.valid_labels))
        width = 0.35
        true_vals = [true_counts[l] for l in self.valid_labels]
        pred_vals = [pred_counts[l] for l in self.valid_labels]
        ax3.bar(x - width/2, true_vals, width, label='True', alpha=0.8)
        ax3.bar(x + width/2, pred_vals, width, label='Predicted', alpha=0.8)
        ax3.set_title('Class Distribution', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.valid_labels, rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=8)
        
        # 4. Overall metrics text
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')
        accuracy = accuracy_score(y_true, y_pred) * 100
        total_samples = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        incorrect = total_samples - correct
        
        metrics_text = f"""
        Overall Performance Metrics
        ═══════════════════════
        Total Samples: {total_samples}
        Correct: {correct}
        Incorrect: {incorrect}
        Accuracy: {accuracy:.2f}%
        
        Macro Avg Precision: {precision.mean():.3f}
        Macro Avg Recall: {recall.mean():.3f}
        Macro Avg F1-Score: {f1.mean():.3f}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 5. Mistakes by class
        ax5 = fig.add_subplot(gs[2, 1:])
        from collections import defaultdict
        
        correct_counts = defaultdict(int)
        incorrect_counts = defaultdict(int)
        for t, p in zip(y_true, y_pred):
            if t == p:
                correct_counts[t] += 1
            else:
                incorrect_counts[t] += 1
        
        x = np.arange(len(self.valid_labels))
        correct_vals = [correct_counts[l] for l in self.valid_labels]
        incorrect_vals = [incorrect_counts[l] for l in self.valid_labels]
        
        ax5.bar(x, correct_vals, label='Correct', color='#2ecc71', alpha=0.8)
        ax5.bar(x, incorrect_vals, bottom=correct_vals, label='Incorrect', 
               color='#e74c3c', alpha=0.8)
        ax5.set_title('Prediction Results by Class', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(self.valid_labels, rotation=45, ha='right', fontsize=8)
        ax5.legend(fontsize=8)
        
        fig.suptitle('IDS Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        return self.save_figure(fig, 'performance_dashboard', 'performance')
    
    # 9. GENERATE ALL FIGURES
    
    def generate_all_figures(
        self,
        y_true: List[str],
        y_pred: List[str],
        predictions_over_time: Optional[List[Tuple[int, str, str, bool]]] = None,
        df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Generate all publication-ready figures
        
        Returns:
            Dictionary mapping figure names to file paths
        """
        self.logger.info("\nGenerating Research Paper Figures...")
        self.logger.separator()
        
        figures = {}
        
        # Confusion Matrices
        self.logger.info("Confusion Matrices...")
        figures['cm_standard'] = self.plot_confusion_matrix_standard(y_true, y_pred)
        figures['cm_normalized'] = self.plot_confusion_matrix_normalized(y_true, y_pred)
        figures['cm_combined'] = self.plot_confusion_matrix_combined(y_true, y_pred)
        
        # Performance Metrics
        self.logger.info("Performance Metrics...")
        figures['per_class_metrics'] = self.plot_per_class_metrics(y_true, y_pred)
        figures['metrics_radar'] = self.plot_metrics_radar(y_true, y_pred)
        figures['mistakes_by_class'] = self.plot_mistakes_by_class(y_true, y_pred)
        
        # Distribution Plots
        self.logger.info("Distribution Plots...")
        figures['class_distribution'] = self.plot_class_distribution(y_true, y_pred)
        figures['class_balance_pie'] = self.plot_class_balance_pie(y_true)
        
        # Learning Curves (if temporal data available)
        if predictions_over_time:
            self.logger.info("Learning Curves...")
            figures['learning_curve'] = self.plot_learning_curve(predictions_over_time)
            figures['error_rate'] = self.plot_error_rate_over_time(predictions_over_time)
        
        # Feature Analysis (if dataframe available)
        if df is not None and feature_columns:
            self.logger.info("Feature Analysis...")
            try:
                figures['feature_correlation'] = self.plot_feature_correlation_heatmap(
                    df, feature_columns
                )
            except Exception as e:
                self.logger.warning(f"Could not generate feature correlation: {e}")
        
        # Comprehensive Dashboard
        self.logger.info("Creating Comprehensive Dashboard...")
        figures['dashboard'] = self.create_performance_dashboard(y_true, y_pred)
        
        self.logger.separator()
        self.logger.info(f"Generated {len(figures)} figures successfully!")
        
        return figures
