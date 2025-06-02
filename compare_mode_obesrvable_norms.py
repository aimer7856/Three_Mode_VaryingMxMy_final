import numpy as np
import os
import re
import matplotlib.pyplot as plt

def extract_mx_my(folder_name):
    match = re.match(r"mx([\d.]+)_my([\d.]+)", folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_observable(path, mode, observable="position"):
    data = np.load(path)
    # Oscillator observables
    if observable in ["osc_position", "osc_momentum", "osc_energy"]:
        if mode == "quantum":
            t = data["t"]
            if observable == "osc_position":
                return t, data["oscillator"][:, 1]
            elif observable == "osc_momentum":
                return t, data["oscillator"][:, 2]
            elif observable == "osc_energy":
                return t, data["oscillator"][:, 5]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        elif mode == "cq":
            t = data["t"]
            if observable == "osc_position":
                return t, data["x"]
            elif observable == "osc_momentum":
                return t, data["px"]
            elif observable == "osc_energy":
                return t, data["Hx"]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        elif mode == "classical":
            t = data["t"]
            if observable == "osc_position":
                return t, data["x"]
            elif observable == "osc_momentum":
                return t, data["px"]
            elif observable == "osc_energy":
                return t, data["Hx"]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    # Projectile observables
    elif observable in ["proj_position", "proj_momentum", "proj_energy"]:
        if mode == "quantum":
            t = data["t"]
            if observable == "proj_position":
                return t, data["projectile"][:, 1]
            elif observable == "proj_momentum":
                return t, data["projectile"][:, 2]
            elif observable == "proj_energy":
                return t, data["projectile"][:, 5]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        elif mode == "cq":
            t = data["t"]
            if observable == "proj_position":
                return t, data["x"]
            elif observable == "proj_momentum":
                return t, data["px"]
            elif observable == "proj_energy":
                return t, data["Hx"]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        elif mode == "classical":
            t = data["t"]
            if observable == "proj_position":
                return t, data["x"]
            elif observable == "proj_momentum":
                return t, data["px"]
            elif observable == "proj_energy":
                return t, data["Hx"]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        elif mode == "classical":
            t = data["t"]
            if observable == "proj_position":
                return t, data["x"]
            elif observable == "proj_momentum":
                return t, data["px"]
            elif observable == "proj_energy":
                return t, data["Hx"]
            else:
                raise ValueError(f"Unknown observable: {observable}")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        raise ValueError(f"Unknown observable: {observable}")

def compute_l2_norm(t, x1, x2):
    dt = np.mean(np.diff(t))
    return np.sqrt(np.sum((x1 - x2)**2) * dt)

def compute_l2_norm_time_series(x1, x2):
    return np.abs(x1 - x2)

def collect_all_norms(root_dir):
    # Oscillator and projectile observables
    osc_observables = ["osc_position", "osc_momentum", "osc_energy"]
    proj_observables = ["proj_position", "proj_momentum", "proj_energy"]
    mode_pairs = [
        ("quantum", "cq"),
        ("quantum", "classical"),
        ("cq", "classical")
    ]
    # Prepare result dictionaries for both sets
    results = {f"{obs}_{m1}_vs_{m2}": [] for obs in osc_observables + proj_observables for m1, m2 in mode_pairs}
    results_time = {f"{obs}_{m1}_vs_{m2}": [] for obs in osc_observables + proj_observables for m1, m2 in mode_pairs}

    subdirs = os.listdir(os.path.join(root_dir, "quantum"))
    for folder in sorted(subdirs):
        mx, my = extract_mx_my(folder)
        if mx is None:
            continue

        base = f"mx{mx}_my{my}"
        try:
            paths = {
                "quantum": os.path.join(root_dir, "quantum", base, f"quantum_{base}.npz"),
                "cq": os.path.join(root_dir, "cq", base, f"cq_{base}.npz"),
                "classical": os.path.join(root_dir, "classical", base, f"classical_{base}.npz"),
            }

            if not all(os.path.exists(p) for p in paths.values()):
                continue

            # Oscillator observables
            for observable in osc_observables:
                data = {mode: load_observable(paths[mode], mode, observable) for mode in paths}
                if not (np.allclose(data["quantum"][0], data["cq"][0]) and
                        np.allclose(data["quantum"][0], data["classical"][0])):
                    continue
                t = data["quantum"][0]
                for m1, m2 in mode_pairs:
                    x1 = data[m1][1]
                    x2 = data[m2][1]
                    key = f"{observable}_{m1}_vs_{m2}"
                    results[key].append((mx, my, compute_l2_norm(t, x1, x2)))
                    results_time[key].append((mx, my, compute_l2_norm_time_series(x1, x2)))

            # Projectile observables
            for observable in proj_observables:
                data = {mode: load_observable(paths[mode], mode, observable) for mode in paths}
                if not (np.allclose(data["quantum"][0], data["cq"][0]) and
                        np.allclose(data["quantum"][0], data["classical"][0])):
                    continue
                t = data["quantum"][0]
                for m1, m2 in mode_pairs:
                    x1 = data[m1][1]
                    x2 = data[m2][1]
                    key = f"{observable}_{m1}_vs_{m2}"
                    results[key].append((mx, my, compute_l2_norm(t, x1, x2)))
                    results_time[key].append((mx, my, compute_l2_norm_time_series(x1, x2)))

        except Exception as e:
            print(f"Error at mx={mx}, my={my}: {e}")

    return results, results_time

def plot_heatmap(data, title, ax=None):
    mx_vals = sorted(set(mx for mx, _, _ in data))
    my_vals = sorted(set(my for _, my, _ in data))
    Z = np.full((len(my_vals), len(mx_vals)), np.nan)

    for mx, my, val in data:
        i = my_vals.index(my)
        j = mx_vals.index(mx)
        Z[i, j] = val

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_fig = True
    else:
        show_fig = False

    im = ax.imshow(Z, origin="lower", extent=[min(mx_vals), max(mx_vals), min(my_vals), max(my_vals)],
                  aspect='auto', cmap='viridis')
    ax.set_xlabel("mx")
    ax.set_ylabel("my")
    ax.set_title(title)
    if show_fig:
        cbar = plt.colorbar(im, ax=ax, label=r"$F(mx, my)$")
        plt.show()
    else:
        return im

# New function for plotting oscillator and projectile heatmaps in separate figures
def plot_observable_heatmaps(results, root_dir, share_colorbar=True):
    # Oscillator observables
    osc_observables = ["osc_position", "osc_momentum", "osc_energy"]
    proj_observables = ["proj_position", "proj_momentum", "proj_energy"]
    mode_pairs = [
        ("quantum", "cq"),
        ("quantum", "classical"),
        ("cq", "classical")
    ]
    mode_labels = {
        "quantum": "Quantum-Quantum",
        "cq": "Classical-Quantum",
        "classical": "Classical-Classical"
    }
    # --- Oscillator Observables Figure ---
    osc_titles = []
    osc_keys = []
    osc_labels = {"osc_position": "Position", "osc_momentum": "Momentum", "osc_energy": "Energy"}
    for obs in osc_observables:
        for m1, m2 in mode_pairs:
            title = f"{osc_labels[obs]}: {mode_labels[m1]} vs {mode_labels[m2]}"
            osc_titles.append(title)
            osc_keys.append(f"{obs}_{m1}_vs_{m2}")

    fig_osc, axes_osc = plt.subplots(3, 3, figsize=(20, 18))
    ims_osc = []
    for ax, key, title in zip(axes_osc.flat, osc_keys, osc_titles):
        im = plot_heatmap(results[key], title, ax=ax)
        ims_osc.append(im)
    fig_osc.suptitle("Oscillator Observable Comparison", fontsize=22, y=1.02)
    plt.tight_layout()
    if share_colorbar:
        fig_osc.subplots_adjust(right=0.92)
        cbar_ax = fig_osc.add_axes([0.94, 0.15, 0.02, 0.7])
        vmin = min(im.get_array().min() for im in ims_osc)
        vmax = max(im.get_array().max() for im in ims_osc)
        norm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        norm.set_array([])
        cbar = fig_osc.colorbar(norm, cax=cbar_ax, label=r"$F(mx, my)$")
    plt.show()
    save_dir = os.path.join(root_dir, "comparison_plots")
    os.makedirs(save_dir, exist_ok=True)
    fig_osc.savefig(os.path.join(save_dir, "comparison_heatmaps_oscillator_observables.png"), bbox_inches='tight')

    # --- Projectile Observables Figure ---
    proj_titles = []
    proj_keys = []
    proj_labels = {"proj_position": "Position", "proj_momentum": "Momentum", "proj_energy": "Energy"}
    for obs in proj_observables:
        for m1, m2 in mode_pairs:
            title = f"{proj_labels[obs]}: {mode_labels[m1]} vs {mode_labels[m2]}"
            proj_titles.append(title)
            proj_keys.append(f"{obs}_{m1}_vs_{m2}")

    fig_proj, axes_proj = plt.subplots(3, 3, figsize=(20, 18))
    ims_proj = []
    for ax, key, title in zip(axes_proj.flat, proj_keys, proj_titles):
        im = plot_heatmap(results[key], title, ax=ax)
        ims_proj.append(im)
    fig_proj.suptitle("Projectile Observable Comparison", fontsize=22, y=1.02)
    plt.tight_layout()
    if share_colorbar:
        fig_proj.subplots_adjust(right=0.92)
        cbar_ax = fig_proj.add_axes([0.94, 0.15, 0.02, 0.7])
        vmin = min(im.get_array().min() for im in ims_proj)
        vmax = max(im.get_array().max() for im in ims_proj)
        norm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        norm.set_array([])
        cbar = fig_proj.colorbar(norm, cax=cbar_ax, label=r"$F(mx, my)$")
    plt.show()
    fig_proj.savefig(os.path.join(save_dir, "comparison_heatmaps_projectile_observables.png"), bbox_inches='tight')

if __name__ == "__main__":
    generate_animations = True  # Set to False to skip animation rendering
    root_dir = "/Users/doyeonkim/OneDrive/Documents/Project1_Sanjeev/Three_Mode_VaryingMxMy_May23/results_Neig32tn4096ny4096"  # CHANGE THIS to your actual root directory
    results, results_time = collect_all_norms(root_dir)
    plot_observable_heatmaps(results, root_dir)

    if generate_animations:
        # Generate heatmap animations of time evolution using results_time
        import matplotlib.animation as animation

        def create_heatmap_animation(time_series_data, title_prefix, save_path, mx_vals, my_vals, max_frames=100):
            fig, ax = plt.subplots(figsize=(8, 6))
            Z = np.full((len(my_vals), len(mx_vals)), np.nan)
            # Compute vmax from the actual data
            vmax = max(val.max() for _, _, val in time_series_data)
            im = ax.imshow(Z, origin="lower", extent=[min(mx_vals), max(mx_vals), min(my_vals), max(my_vals)],
                           aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
            ax.set_xlabel("mx")
            ax.set_ylabel("my")
            title = ax.set_title("")
            cbar = plt.colorbar(im, ax=ax, label=r"$F(mx, my)$")

            # Determine uniform sampling frames
            T = min(len(val) for _, _, val in time_series_data)
            indices = np.linspace(0, T - 1, min(max_frames, T), dtype=int)

            def update(frame_idx):
                Z[:, :] = np.nan
                frame = indices[frame_idx]
                for mx, my, val in time_series_data:
                    i = my_vals.index(my)
                    j = mx_vals.index(mx)
                    Z[i, j] = val[frame] if frame < len(val) else np.nan
                im.set_array(Z)
                title.set_text(f"{title_prefix} - Frame {frame}")
                return [im, title]

            ani = animation.FuncAnimation(fig, update, frames=len(indices), interval=200, blit=False)
            ani.save(save_path, writer='ffmpeg', dpi=150)
            plt.close()

        # Run animation creation for all keys
        all_keys = list(results_time.keys())
        mx_vals_all = sorted(set(mx for key in all_keys for mx, _, _ in results_time[key]))
        my_vals_all = sorted(set(my for key in all_keys for _, my, _ in results_time[key]))
        save_dir = os.path.join(root_dir, "comparison_plots")

        for key in all_keys:
            parts = key.split("_")
            obs = "_".join(parts[:-3])
            m1 = parts[-3]
            m2 = parts[-1]
            label = " ".join(obs.split("_")).capitalize()
            title_prefix = f"{label} {m1} vs {m2}"
            save_path = os.path.join(save_dir, f"{key}_evolution.mp4")
            create_heatmap_animation(results_time[key], title_prefix, save_path, mx_vals_all, my_vals_all)