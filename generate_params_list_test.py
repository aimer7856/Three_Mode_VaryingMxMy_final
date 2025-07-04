import itertools

#import os

#os.makedirs("test",exist_ok=True)

# Parameters to sweep
modes       = ["qq", "cq", "cc"]
mx_vals     = [1.0]
my_vals     = [1.0]
x0_vals     = [0.0, 1.0, 2.0]
vx0_vals    = [0.0]

# Fixed parameters
nx_vals     = [32]
xmin_vals   = [-5.0]
xmax_vals   = [5.0]
ny_vals     = [128]
ymin_vals   = [-5.0]
ymax_vals   = [64.0]
y0_vals     = [10.0]
vy0_vals    = [-1.0]
sigmay_vals = [3.0]
total_time_vals = [10.0]
timesteps_vals  = [128]
lambda_vals = [0.0]
n_eig_vals  = [16]

# Header
header = [
    "mode", "mx", "my", "x0", "vx0",
    "nx", "xmin", "xmax", "ny", "ymin", "ymax",
    "y0", "vy0", "sigmay", "total_time", "timesteps",
    "lambda", "n_eig", "filename"
]

#with open(os.path.join("test", "coherent_param_list.txt"), "w") as f:
with open("coherent_param_list.txt", "w") as f:
    f.write(','.join(header) + '\n')
    for combo in itertools.product(
        modes, mx_vals, my_vals, x0_vals, vx0_vals,
        nx_vals, xmin_vals, xmax_vals, ny_vals, ymin_vals, ymax_vals,
        y0_vals, vy0_vals, sigmay_vals, total_time_vals, timesteps_vals,
        lambda_vals, n_eig_vals
    ):
        (
            mode, mx, my, x0, vx0,
            nx, xmin, xmax, ny, ymin, ymax,
            y0, vy0, sigmay, total_time, timesteps,
            lambda_, n_eig
        ) = combo

        # Only include varying parameters in filename
        filename_parts = []
        if len(mx_vals) > 1:
            filename_parts.append(f"mx{mx}")
        if len(my_vals) > 1:
            filename_parts.append(f"my{my}")
        if len(modes) > 1:
            filename_parts.insert(0, mode)  # put mode first

        filename = "_".join(filename_parts)
        # Add x0 or vx0 to filename if they are varied
        if len(x0_vals) > 1:
            filename_parts.append(f"x0{x0}")
        if len(vx0_vals) > 1:
            filename_parts.append(f"vx0{vx0}")
        row = list(map(str, combo)) + [filename]
        f.write(','.join(row) + '\n')