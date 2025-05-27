import numpy as np
import os
import re
import matplotlib.pyplot as plt

def extract_mx_my(folder_name):
    match = re.match(r"mx([\d.]+)_my([\d.]+)", folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_observable(path, mode):
    data = np.load(path)
    if mode == "quantum":
        return data["t"], data["oscillator"][:, 0]
    elif mode == "cq":
        return data["t"], data["x"]
    elif mode == "classical":
        return data["t"], data["x"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def compute_l2_norm(t, x1, x2):
    dt = np.mean(np.diff(t))
    return np.sqrt(np.sum((x1 - x2)**2) * dt)

def collect_all_norms(root_dir):
    results = {
        "quantum_vs_cq": [],
        "quantum_vs_classical": [],
        "cq_vs_classical": []
    }

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

            data = {mode: load_observable(paths[mode], mode) for mode in paths}
            if not (np.allclose(data["quantum"][0], data["cq"][0]) and
                    np.allclose(data["quantum"][0], data["classical"][0])):
                continue

            t = data["quantum"][0]
            qx, cx, kx = data["quantum"][1], data["cq"][1], data["classical"][1]

            results["quantum_vs_cq"].append((mx, my, compute_l2_norm(t, qx, cx)))
            results["quantum_vs_classical"].append((mx, my, compute_l2_norm(t, qx, kx)))
            results["cq_vs_classical"].append((mx, my, compute_l2_norm(t, cx, kx)))

        except Exception as e:
            print(f"Error at mx={mx}, my={my}: {e}")

    return results

def plot_heatmap(data, title):
    mx_vals = sorted(set(mx for mx, _, _ in data))
    my_vals = sorted(set(my for _, my, _ in data))
    Z = np.full((len(my_vals), len(mx_vals)), np.nan)

    for mx, my, val in data:
        i = my_vals.index(my)
        j = mx_vals.index(mx)
        Z[i, j] = val

    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin="lower", extent=[min(mx_vals), max(mx_vals), min(my_vals), max(my_vals)],
               aspect='auto', cmap='viridis')
    plt.colorbar(label=r"$F(mx, my)$")
    plt.xlabel("mx")
    plt.ylabel("my")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    root_dir = "/path/to/root"  # CHANGE THIS to your actual root directory
    results = collect_all_norms(root_dir)

    plot_heatmap(results["quantum_vs_cq"], "Quantum vs CQ")
    plot_heatmap(results["quantum_vs_classical"], "Quantum vs Classical")
    plot_heatmap(results["cq_vs_classical"], "CQ vs Classical")