"""
Knowledge Distillation Training Monitor.
Tracks loss curves, metrics, timing, and provides visualization.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math
import traceback

# Matplotlib for post-epoch plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrainingMonitor:
    """Monitor training progress with detailed metrics."""
    
    def __init__(self, checkpoint_dir: str = "kd_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss tracking
        self.step_losses = []  # Per-step losses
        self.epoch_losses = []  # Per-epoch average losses
        self.epoch_min_losses = []  # Min loss per epoch
        
        # Timing tracking
        self.step_times = []  # Time per step
        self.epoch_start_time = None
        self.epoch_times = []  # Time per epoch
        self.training_start_time = None
        
        # Step tracking
        self.global_step = 0
        self.current_epoch = 0
        
        # Metrics
        self.metrics_history = {
            "step": [],
            "loss": [],
            "avg_loss": [],
            "epoch": [],
            "timestamp": [],
            "time_elapsed": [],
            "time_per_step": [],
        }
            
    def start_training(self):
        """Mark start of training."""
        self.training_start_time = time.time()
    
    def start_epoch(self, epoch: int):
        """Mark start of epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def log_step(self, step: int, loss: float):
        """Log a training step."""
        self.global_step = step
        self.step_losses.append(loss)

        # Calculate metrics
        avg_loss = sum(self.step_losses) / len(self.step_losses)
        time_elapsed = time.time() - self.training_start_time
        time_per_step = time_elapsed / len(self.step_losses)

        # Record in history
        self.metrics_history["step"].append(step)
        self.metrics_history["loss"].append(loss)
        self.metrics_history["avg_loss"].append(avg_loss)
        self.metrics_history["epoch"].append(self.current_epoch)
        self.metrics_history["timestamp"].append(datetime.now().isoformat())
        self.metrics_history["time_elapsed"].append(time_elapsed)
        self.metrics_history["time_per_step"].append(time_per_step)


    def end_epoch(self, epoch_avg_loss: float, min_loss: Optional[float] = None):
        """Mark end of epoch."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.epoch_losses.append(epoch_avg_loss)
        
        if min_loss is not None:
            self.epoch_min_losses.append(min_loss)
        
        # Plot after epoch
        self.plot_loss_curve(epoch=self.current_epoch)
    
    def log_error(self, error_msg: str):
        """Log error to text file."""
        log_path = self.checkpoint_dir / "training_errors.log"
        with open(log_path, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"[{datetime.now().isoformat()}] ERROR\n")
            f.write(f"{'='*70}\n")
            f.write(error_msg)
            f.write(f"\n{'='*70}\n")
    
    def plot_loss_curve(self, epoch: int = 0):
        """Plot loss curve and save to file."""
        if not MATPLOTLIB_AVAILABLE or not self.step_losses:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Knowledge Distillation Training - Epoch {epoch + 1}', fontsize=16, fontweight='bold')
            
            # Plot 1: Loss per step
            ax = axes[0, 0]
            ax.plot(self.step_losses, linewidth=1.5, color='#2E86AB')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Loss per Step')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Average loss (cumulative)
            ax = axes[0, 1]
            avg_losses = []
            for i in range(len(self.step_losses)):
                avg_losses.append(sum(self.step_losses[:i+1]) / (i+1))
            ax.plot(avg_losses, linewidth=1.5, color='#A23B72')
            ax.set_xlabel('Step')
            ax.set_ylabel('Average Loss')
            ax.set_title('Cumulative Average Loss')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Loss per epoch
            ax = axes[1, 0]
            if self.epoch_losses:
                ax.bar(range(len(self.epoch_losses)), self.epoch_losses, color='#F18F01', alpha=0.8)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Average Loss')
                ax.set_title('Average Loss per Epoch')
                ax.set_xticks(range(len(self.epoch_losses)))
                ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Time per epoch
            ax = axes[1, 1]
            if self.epoch_times:
                ax.bar(range(len(self.epoch_times)), [t/60 for t in self.epoch_times], color='#06A77D', alpha=0.8)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Time (minutes)')
                ax.set_title('Training Time per Epoch')
                ax.set_xticks(range(len(self.epoch_times)))
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save figure
            plot_path = self.checkpoint_dir / f"loss_curve_epoch_{epoch}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Plot saved: {plot_path.name}")
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not generate plot: {e}")
    
    def save_training_log(self):
        """Save training summary to text file."""
        log_path = self.checkpoint_dir / "training_summary.log"
        
        stats = self.get_stats()
        trend = self.get_loss_trend(last_n=len(self.step_losses))
        
        with open(log_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("KNOWLEDGE DISTILLATION TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Training Time: {stats.get('time_elapsed_formatted', 'N/A')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("LOSS METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Current Loss:       {stats.get('current_loss', 'N/A'):.6f}\n")
            f.write(f"Average Loss:       {stats.get('avg_loss', 'N/A'):.6f}\n")
            f.write(f"Min Loss:           {stats.get('min_loss', 'N/A'):.6f}\n")
            f.write(f"Max Loss:           {stats.get('max_loss', 'N/A'):.6f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("EPOCH SUMMARY\n")
            f.write("-" * 70 + "\n")
            for i, (loss, time_sec) in enumerate(zip(self.epoch_losses, self.epoch_times)):
                f.write(f"Epoch {i+1}: Avg Loss = {loss:.6f}, Time = {self._format_time(time_sec)}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("STEP INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Steps:        {stats.get('total_steps', 0)}\n")
            f.write(f"Total Epochs:       {len(self.epoch_losses)}\n")
            f.write(f"Avg Time per Step:  {stats.get('avg_time_per_step', 0):.4f}s\n\n")
            
            if trend:
                f.write("-" * 70 + "\n")
                f.write("LOSS TREND (Last 100 steps)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Initial Loss:       {trend.get('initial_loss', 'N/A'):.6f}\n")
                f.write(f"Current Loss:       {trend.get('current_loss', 'N/A'):.6f}\n")
                f.write(f"Improvement:        {trend.get('improvement', 0):.6f} ({trend.get('improvement_percent', 0):.2f}%)\n\n")
            
            f.write("=" * 70 + "\n")
        
        return log_path
    
    def get_stats(self) -> Dict:
        """Get current training statistics."""
        if not self.step_losses:
            return {}
        
        current_loss = self.step_losses[-1]
        avg_loss = sum(self.step_losses) / len(self.step_losses)
        min_loss = min(self.step_losses)
        max_loss = max(self.step_losses)
        
        time_elapsed = time.time() - self.training_start_time if self.training_start_time else 0
        
        stats = {
            "current_step": self.global_step,
            "current_epoch": self.current_epoch,
            "current_loss": current_loss,
            "avg_loss": avg_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "total_steps": len(self.step_losses),
            "time_elapsed": time_elapsed,
            "time_elapsed_formatted": self._format_time(time_elapsed),
            "avg_time_per_step": time_elapsed / len(self.step_losses) if self.step_losses else 0,
        }
        
        if self.epoch_times:
            stats["avg_time_per_epoch"] = sum(self.epoch_times) / len(self.epoch_times)
            stats["last_epoch_time"] = self.epoch_times[-1]
        
        return stats
    
    def get_loss_trend(self, last_n: int = 100) -> Dict:
        """Get loss trend for last N steps."""
        losses = self.step_losses[-last_n:] if len(self.step_losses) > last_n else self.step_losses
        
        if not losses:
            return {}
        
        trend = {
            "initial_loss": losses[0],
            "current_loss": losses[-1],
            "improvement": losses[0] - losses[-1],
            "improvement_percent": ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0,
            "min_loss": min(losses),
            "max_loss": max(losses),
            "avg_loss": sum(losses) / len(losses),
            "steps_analyzed": len(losses),
        }
        
        return trend
    
    def get_time_estimate(self, total_steps: int) -> Dict:
        """Estimate remaining time."""
        if not self.step_times or self.global_step == 0:
            return {"estimated_remaining": "N/A", "estimated_total": "N/A"}
        
        avg_time_per_step = sum(self.step_times) / len(self.step_times)
        remaining_steps = total_steps - self.global_step
        estimated_remaining = avg_time_per_step * remaining_steps
        
        elapsed = time.time() - self.training_start_time if self.training_start_time else 0
        estimated_total = elapsed + estimated_remaining
        
        return {
            "avg_time_per_step": avg_time_per_step,
            "remaining_steps": remaining_steps,
            "estimated_remaining": self._format_time(estimated_remaining),
            "estimated_total": self._format_time(estimated_total),
            "elapsed": self._format_time(elapsed),
        }
    
    def save_metrics(self, filename: str = "metrics.json"):
        """Save all metrics to JSON."""
        filepath = self.checkpoint_dir / filename
        
        metrics = {
            "metadata": {
                "training_start": datetime.now().isoformat() if self.training_start_time else None,
                "total_steps": len(self.step_losses),
                "total_epochs": len(self.epoch_losses),
                "total_time": time.time() - self.training_start_time if self.training_start_time else 0,
            },
            "step_metrics": self.metrics_history,
            "epoch_summary": {
                "epoch_losses": self.epoch_losses,
                "epoch_times": self.epoch_times,
                "epoch_min_losses": self.epoch_min_losses,
            },
            "final_stats": self.get_stats(),
            "loss_trend": self.get_loss_trend(last_n=len(self.step_losses)),
        }
        
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return filepath
    
    def save_csv(self, filename: str = "training_log.csv"):
        """Save detailed metrics to CSV for plotting."""
        filepath = self.checkpoint_dir / filename
        
        with open(filepath, "w") as f:
            # Header
            f.write("step,epoch,loss,avg_loss,time_elapsed,time_per_step,timestamp\n")
            
            # Data
            for i in range(len(self.step_losses)):
                step = self.metrics_history["step"][i]
                epoch = self.metrics_history["epoch"][i]
                loss = self.metrics_history["loss"][i]
                avg_loss = self.metrics_history["avg_loss"][i]
                time_elapsed = self.metrics_history["time_elapsed"][i]
                time_per_step = self.metrics_history["time_per_step"][i]
                timestamp = self.metrics_history["timestamp"][i]
                
                f.write(f"{step},{epoch},{loss:.6f},{avg_loss:.6f},{time_elapsed:.2f},{time_per_step:.6f},{timestamp}\n")
        
        return filepath
    
    def print_summary(self):
        """Print training summary."""
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        
        print(f"\nLoss Metrics:")
        current_loss = stats.get('current_loss', 'N/A')
        print(f"  Current Loss:     {current_loss:.6f if isinstance(current_loss, (int, float)) else current_loss}")
        avg_loss = stats.get('avg_loss', 'N/A')
        print(f"  Average Loss:     {avg_loss:.6f if isinstance(avg_loss, (int, float)) else avg_loss}")
        min_loss = stats.get('min_loss', 'N/A')
        print(f"  Min Loss:         {min_loss:.6f if isinstance(min_loss, (int, float)) else min_loss}")
        max_loss = stats.get('max_loss', 'N/A')
        print(f"  Max Loss:         {max_loss:.6f if isinstance(max_loss, (int, float)) else max_loss}")
        
        print(f"\nStep/Epoch Info:")
        print(f"  Total Steps:      {stats.get('total_steps', 0)}")
        print(f"  Current Epoch:    {stats.get('current_epoch', 0)}")
        
        print(f"\nTiming:")
        print(f"  Total Time:       {stats.get('time_elapsed_formatted', 'N/A')}")
        avg_time = stats.get('avg_time_per_step', 0)
        print(f"  Avg Time/Step:    {avg_time:.4f if isinstance(avg_time, (int, float)) else avg_time}s")
        
        if "last_epoch_time" in stats:
            print(f"  Last Epoch Time:  {self._format_time(stats['last_epoch_time'])}")
        
        trend = self.get_loss_trend()
        if trend:
            print(f"\nLoss Trend (Last 100 steps):")
            print(f"  Initial Loss:     {trend.get('initial_loss', 'N/A'):.6f}")
            print(f"  Current Loss:     {trend.get('current_loss', 'N/A'):.6f}")
            print(f"  Improvement:      {trend.get('improvement', 0):.6f} ({trend.get('improvement_percent', 0):.2f}%)")
        
        print("\n" + "=" * 70 + "\n")
        
        # Also save to file
        self.save_training_log()
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        if seconds < 0:
            return "N/A"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class LossVisualizer:
    """Helper class for generating loss curve visualizations."""
    
    @staticmethod
    def generate_plot_data(monitor: TrainingMonitor, output_file: str = "loss_curve.txt"):
        """Generate ASCII plot of loss curve."""
        if not monitor.step_losses:
            return
        
        losses = monitor.step_losses
        
        # Normalize losses for display
        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss if max_loss > min_loss else 1
        
        # Chart dimensions
        width = 80
        height = 20
        
        # Create chart
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        step_interval = max(1, len(losses) // width)
        for i in range(0, len(losses), step_interval):
            x = min(int(i / step_interval), width - 1)
            # Normalize loss to height
            normalized = (losses[i] - min_loss) / loss_range if loss_range > 0 else 0.5
            y = height - 1 - int(normalized * (height - 1))
            y = max(0, min(y, height - 1))
            chart[y][x] = '●'
        
        # Print chart
        output = "Loss Curve:\n"
        output += "┌" + "─" * (width - 2) + "┐\n"
        for row in chart:
            output += "│" + "".join(row) + "│\n"
        output += "└" + "─" * (width - 2) + "┘\n"
        output += f"Max: {max_loss:.6f}  Min: {min_loss:.6f}\n"
        output += f"Steps: {len(losses)}\n"
        
        return output
