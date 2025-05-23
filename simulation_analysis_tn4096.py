import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

def extract_mx_my(folder_name):
    """
    Extract mx and my values from folder names like mx1.0_my0.5
    """
    match = re.match(r"mx(\d+(?:\.\d+)?)_my(\d+(?:\.\d+)?)", folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_simulation_data(folder, mode):
    """
    Load .npz + params.json from a folder, according to simulation mode.
    mode must be one of: 'quantum', 'classical', 'cq'
    """
    if mode == "quantum":
        npz_files = glob.glob(os.path.join(folder, "*_*.npz"))
    elif mode == "classical":
        npz_files = glob.glob(os.path.join(folder, "*classical_*.npz"))
    elif mode == "cq":
        npz_files = glob.glob(os.path.join(folder, "*cq_*.npz"))
    else:
        raise ValueError(f"Unknown mode {mode}")

    json_files = glob.glob(os.path.join(folder, "*_params*.json"))

    if not npz_files or not json_files:
        raise FileNotFoundError(f"Missing data for {mode} in {folder}")

    data = np.load(npz_files[0])
    with open(json_files[0], "r") as f:
        params = json.load(f)
    return data, params

def scan_all_data(root_dir="results_tn4096"):
    """
    Scan root_dir for subfolders: 'quantum', 'cq', 'classiscal'
    Under each, look for mx*_my* folders and load data.
    Returns:
        results[(mx, my)] = {'quantum':(...), 'classical':(...), 'cq':(...)}
    """
    folder_map = {
        "quantum": "quantum",
        "cq":      "cq",
        "classical": "classical"
    }
    results = {}
    for folder_name, mode_key in folder_map.items():
        mode_dir = os.path.join(root_dir, folder_name)
        if not os.path.isdir(mode_dir):
            continue
        for sub in os.listdir(mode_dir):
            subpath = os.path.join(mode_dir, sub)
        
            if not os.path.isdir(subpath):
                continue
            mx, my = extract_mx_my(sub)
        
            if mx is None or my is None:
                continue
            try:
                #print("Contents of", subpath, ":", os.listdir(subpath))
                #print("  .npz matches:", glob.glob(os.path.join(subpath, "*_*.npz")))
                #print("  .json matches:", glob.glob(os.path.join(subpath, "*params_*.json")))
                data, params = load_simulation_data(subpath, mode_key)
                results.setdefault((mx, my), {})[mode_key] = (data, params)
            except Exception as e:
                print(f"[WARN] {mode_key} load failed at {subpath}: {e}")
    return results

def plot_subplot(ax, title, label, t, data_c, data_q, data_cq=None, std_q=None):
    ax.plot(t, data_c, 'C0-', label='Classical-Classical')
    ax.plot(t, data_q, 'C1-', label='Quantum-Quantum')
    if data_cq is not None:
        ax.plot(t, data_cq, 'C2-', label='Classical-Quantum')
    if std_q is not None:
        ax.fill_between(t, data_q - std_q, data_q + std_q,color='C1', alpha=0.3, label='Quantum ±1σ')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True)

def process_folder(mx, my, data_dict, out_dir):
    fig, axs = plt.subplots(3,3,figsize=(18,18))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.88, wspace=0.15, hspace=0.2)

    # Initialize variables
    rho1_init = x_grid = None
    int_c = std_int_q = None

    # Quantum data
    if "quantum" in data_dict:
        qdata, qparams = data_dict["quantum"]
        t = qdata["t"]
        osc = qdata["oscillator"]
        proj = qdata["projectile"]
        exp_x, std_x   = osc[:,1], osc[:,3]
        exp_px, std_px = osc[:,2], osc[:,4]
        Hx, std_Hx     = osc[:,5], osc[:,6]
        exp_y, std_y   = proj[:,1], proj[:,3]
        exp_py, std_py = proj[:,2], proj[:,4]
        Hy, std_Hy     = proj[:,5], proj[:,6]
        vn = qdata["vn_entropy"]
        lin = qdata.get("linear_entropy", None)
        rho1_init = qdata["rho1_diag"][0]
        int_q = qdata["inter_energy"]
        std_int_q = qdata.get("std_inter_energy", None)
        
        qmeta = qparams.get("quantum", {})
        xmin = qmeta.get('xmin', -10)
        xmax = qmeta.get('xmax', 10)
        nx   = qmeta.get('nx', 256)
        x0   = qmeta.get('x0', 0.0)
        sigmax = qmeta.get('sigmax', 1.0)
        my   = qmeta.get('my', 1)
        ymin = qmeta.get('ymin', -10)
        ymax = qmeta.get('ymax', 100)
        ny   = qmeta.get('ny', 2048)
        y0   = qmeta.get('y0', 0.0)
        sigmay = qmeta.get('sigmay', 3.0)
        
        lambda_ = qmeta.get('lambda')
        total_time = qmeta.get('total_time')
        timesteps = qmeta.get('timesteps')
        
        qruntime = qmeta.get('runtime_quantum')
        
        x_grid = np.linspace(xmin, xmax, nx)

    # Classical data (folder 'classiscal')
    if "classical" in data_dict:
        cdata, cparams = data_dict["classical"]
        cc_x, cc_px, cc_Hx = cdata["x"], cdata["px"], cdata["Hx"]
        cc_y, cc_py, cc_Hy = cdata["y"], cdata["py"], cdata["Hy"]
        int_c = cdata["Hint"]
        t_c = cdata["t"] 
        
        cmeta = cparams.get("classical", {})  # dive into inner dict
        cruntime = cmeta.get('runtime_classical')
        
    # CQ data
    if "cq" in data_dict:
        cqdata, cqparams = data_dict["cq"]
        cq_x, cq_px, cq_Hx = cqdata["x"], cqdata["px"], cqdata["Hx"]
        cq_y, cq_py, cq_Hy = cqdata["y"], cqdata["py"], cqdata["Hy"]
        
        cqmeta = cqparams.get("cq", {})  # dive into inner dict
        N_eig = cqmeta.get('N_eig')
        cqruntime = cqmeta.get('runtime_cq')
       # print(cqruntime)
        
    # Row 1: Oscillator
    osc_items = [
        ('Oscillator Position ⟨x⟩', '⟨x⟩',  exp_x, std_x, cc_x, cq_x),
        ('Oscillator Momentum ⟨px⟩', '⟨px⟩', exp_px, std_px, cc_px, cq_px),
        ('Oscillator Energy ⟨Hx⟩','⟨Hx⟩',   Hx, std_Hx, cc_Hx, cq_Hx)
    ]
    for i, (ttl, lbl, qd, std_q, cd, cqd) in enumerate(osc_items):
        plot_subplot(axs[0,i], ttl, lbl, t, cd, qd, cqd, std_q)

    # Row 2: Projectile
    proj_items = [
        ('Projectile Position ⟨y⟩', '⟨y⟩', exp_y, std_y, cc_y, cq_y),
        ('Projectile Momentum ⟨py⟩', '⟨py⟩', exp_py, std_py, cc_py, cq_py),
        ('Projectile Energy ⟨Hy⟩',  '⟨Hy⟩', Hy, std_Hy, cc_Hy, cq_Hy)
    ]
    for i, (ttl, lbl, qd, std_q, cd, cqd) in enumerate(proj_items):
        plot_subplot(axs[1,i], ttl, lbl, t, cd, qd, cqd, std_q)

    # Row 3, Col 0: Initial profile
    if rho1_init is not None and x_grid is not None:
        ax = axs[2,0]
        analytic = (1/(np.sqrt(2*np.pi)*sigmax)) * np.exp(-(x_grid-x0)**2/(2*sigmax**2))
        ax.plot(x_grid, rho1_init, 'C1--', marker='D',  markevery=1, label='Initial Data')
        ax.plot(x_grid, analytic, 'C2-', linewidth =2, label='Analytic Gaussian')
        ax.set_title('Initial x-profile'); ax.set_xlabel('x'); ax.set_ylabel('Probability density')
        ax.legend(); ax.grid(True)

    # Row 3, Col 1: Interaction Energy
    if int_c is not None:
        plot_subplot(axs[2,1], "Interaction Energy", "Energy", t, int_c, int_q, None, std_int_q)

    # Row 3, Col 2: Entropies
    ax = axs[2,2]
    if "quantum" in data_dict:
        ax.plot(t, vn, 'C2-', label='Von Neumann')
        if lin is not None:
            ax.plot(t, lin, 'C3-', label='Linear')
        ax.set_title('Entropies'); ax.set_xlabel('Time'); ax.set_ylabel('Entropy')
        ax.legend(); ax.grid(True)

    fig = ax.figure
    fig.suptitle(f"QQ vs CC vs CQ Observables (mx,my)= ({mx},{my})", fontsize=18)
    subtitle = "\n".join([
        f"(x0,y0)=({x0},{y0}), (sigmax,sigmay)=({sigmax:.2f}, {sigmay})",
        f"(xmin,xmax)=({xmin},{xmax}), (ymin,ymax)=({ymin},{ymax}), (nx,ny)=({nx},{ny})",
        f"lambda = {lambda_}, N_eig = {N_eig}, total_time = {total_time}, timesteps = {timesteps}",
        f"runtime: QQruntime={qruntime}, CCruntime={cruntime}, CQruntime={cqruntime}"
    ])
    fig.text(0.5, 0.91, subtitle, ha='center', fontsize=14)
    fname = os.path.join(out_dir, f"panel_mx{mx}_my{my}.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)

def plot_entropy_energy_by_mx(grouped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for mx, items in grouped.items():
       
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Energy
        ax_h = axs[0]
        for my,(t,vn,energy) in sorted(items, key=lambda x: x[0]):
            ax_h.plot(t, energy, label=f"my={my}")
        ax_h.set_title(f"Oscillator Energy⟨Hx⟩ vs Time (mx={mx})")
        ax_h.set_xlabel("Time"); ax_h.set_ylabel("⟨Hx⟩"); ax_h.legend(); ax_h.grid(True)
        
        # Entropy
        ax_s = axs[1]
        for my,(t,vn,energy) in sorted(items, key=lambda x: x[0]):
            ax_s.plot(t, vn, label=f"my={my}")
        ax_s.set_title(f"Von Neumann Entory vs Time (mx={mx})")
        ax_s.set_xlabel("Time"); ax_s.set_ylabel("VN Entropy"); ax_s.legend(); ax_s.grid(True)
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"energy_entropy_vs_my_mx{mx}.png"), dpi =300)
        plt.close(fig)

def main():
    root = "results_tn4096"
    panel_dir = os.path.join(root, "panels_all_modes")
    summary_dir = os.path.join(root, "entropy_energy_by_mx")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    all_data = scan_all_data(root_dir=root)
    grouped = {}

    for (mx, my), data_dict in all_data.items():
        try:
            process_folder(mx, my, data_dict, panel_dir)
            if "quantum" in data_dict:
                qdata, _ = data_dict["quantum"]
                t = qdata["t"]; vn = qdata["vn_entropy"]
                energy = qdata["oscillator"][:,5]
                grouped.setdefault(mx, []).append((my,(t,vn,energy)))
        except Exception as e:
            print(f"[ERROR] mx={mx}, my={my} failed: {e}")

    plot_entropy_energy_by_mx(grouped, summary_dir)

if __name__ == "__main__":
    main()
