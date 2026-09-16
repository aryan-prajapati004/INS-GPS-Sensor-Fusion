import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Make sure we can find the models and the artifacts
os.environ["ARTIFACT_DIR"] = os.path.abspath(os.path.join("backend", "artifacts"))
sys.path.append(os.path.abspath('backend'))

# Import existing backend modules
from backend.main import _load_dataset, _load_artifact
from backend.models import WINDOW, STRIDE, MODEL_PALETTE

# Configure Seaborn for beautiful matplotlib plots
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def animate_models_trajectory(model_names, dataset_filename="v1.csv"):
    print(f"Loading dataset {dataset_filename}...")
    raw, gt_PN, gt_PE, dt_arr = _load_dataset(dataset_filename)

    # Build sliding windows
    print("Building windows...")
    windows = []
    gt_pn = []
    gt_pe = []
    dt_win = []
    
    for i in range(0, len(raw) - WINDOW + 1, STRIDE):
        windows.append(raw[i:i + WINDOW])
        idx = i + WINDOW - 1
        gt_pn.append(gt_PN[idx])
        gt_pe.append(gt_PE[idx])
        dt_win.append(dt_arr[idx])
        
    n = len(windows)
    print(f"Total windows: {n}")
    
    # Store predictions for animation
    model_predictions = {}
    
    # Run inference for each model
    for model_name in model_names:
        print(f"Running inference for {model_name}...")
        try:
            entry = _load_artifact(model_name)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue
            
        model_nn, artifact = entry["model"], entry["artifact"]
        f_scaler, t_scaler = artifact["f_scaler"], artifact["t_scaler"]
        stride_val = artifact.get("stride", STRIDE)
        
        all_pred = []
        batch_sz = 512
        for bi in range(0, n, batch_sz):
            batch_raw = windows[bi:bi + batch_sz]
            batch = np.stack([f_scaler.transform(np.asarray(w, dtype=np.float32)) for w in batch_raw])
            x_t = torch.from_numpy(batch).float()
            
            with torch.no_grad():
                pred_scaled = model_nn(x_t).numpy()
            
            all_pred.append(t_scaler.inverse_transform(pred_scaled))
            
        pred_vel = np.concatenate(all_pred, axis=0)
        
        # Pure integration (no reset to GT) to see raw accumulated drift
        pred_PN_arr = np.zeros(n)
        pred_PE_arr = np.zeros(n)
        pred_PN_arr[0] = gt_pn[0]
        pred_PE_arr[0] = gt_pe[0]
        
        for t in range(1, n):
            step_dt = stride_val * dt_win[t]
            pred_PN_arr[t] = pred_PN_arr[t - 1] + pred_vel[t, 0] * step_dt
            pred_PE_arr[t] = pred_PE_arr[t - 1] + pred_vel[t, 1] * step_dt
            
        model_predictions[model_name] = {
            'PE': pred_PE_arr,
            'PN': pred_PN_arr,
            'color': MODEL_PALETTE.get(model_name, "#888888")
        }

    # Setup animation plot
    fig, ax = plt.subplots()
    
    # Plot start point
    if len(gt_pn) > 0:
        ax.scatter([gt_pe[0]], [gt_pn[0]], color='blue', s=150, zorder=10, label='Start', edgecolors='white', linewidths=2)

    ax.set_title("Animated AUV Trajectory (Pure Dead-Reckoning Drift)", fontsize=16, pad=15)
    ax.set_xlabel("East Position (PE) [meters]", fontsize=14)
    ax.set_ylabel("North Position (PN) [meters]", fontsize=14)
    ax.axis('equal')
    
    # Calculate bounds for consistent limits based on max drift
    all_pe = gt_pe.copy()
    all_pn = gt_pn.copy()
    for m_data in model_predictions.values():
        all_pe.extend(m_data['PE'].tolist())
        all_pn.extend(m_data['PN'].tolist())
        
    ax.set_xlim(min(all_pe) - 200, max(all_pe) + 200)
    ax.set_ylim(min(all_pn) - 200, max(all_pn) + 200)
    
    # Initialize empty lines for animation
    # GT gets a thicker, semi-transparent black line
    gt_line, = ax.plot([], [], color='black', linewidth=4, alpha=0.5, label='Ground Truth')
    
    model_lines = {}
    for model_name, m_data in model_predictions.items():
        # Models get vivid colored lines
        line, = ax.plot([], [], color=m_data['color'], linewidth=2.5, label=model_name)
        model_lines[model_name] = line
        
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, title="Models")
    plt.tight_layout()

    # Animation update function
    frames = 200 # We will draw the entire path over 200 frames
    step_size = max(1, n // frames)
    
    def update(frame):
        # Current index to draw up to
        idx = min((frame + 1) * step_size, n)
        
        # Update ground truth line
        gt_line.set_data(gt_pe[:idx], gt_pn[:idx])
        
        # Update model lines
        for m_name, line in model_lines.items():
            m_data = model_predictions[m_name]
            line.set_data(m_data['PE'][:idx], m_data['PN'][:idx])
            
        return [gt_line] + list(model_lines.values())

    print("Starting animation window...")
    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=True, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    # You can change the list of models here
    animate_models_trajectory(["RNN", "LSTM", "BERT"], dataset_filename="v1.csv")
