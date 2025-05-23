#!/bin/bash
#SBATCH --job-name=qq_ny4096
#SBATCH --array=0                    # 3 modes × 4 mx × 4 my = 48 jobs
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=3
#SBATCH --output=logs_qq_ny4096/%x_%A_%a.out
#SBATCH --error=logs_qq_ny4096/%x_%A_%a.err
#SBATCH --mail-user=doyeon.k@unb.ca
#SBATCH --mail-type=ALL

# Make sure SLURM can write its own logs:
mkdir -p logs_qq_ny4096

#  Conda activation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310_env

# Clean environment
unset PYTHONPATH
export PYTHONNOUSERSITE=True



# Parameter sweeps
MODES=("quantum")
MX_VALUES=(0.5)
MY_VALUES=(0.5)

# Fixed parameters
NX=256
XMIN=-10.0
XMAX=10.0
NY=4096
YMIN=-15.0
YMAX=130.0
X0=0.0
Y0=20.0
VX0=0.0
VY0=-5.0
SIGMAY=3.0
TOTAL_TIME=30.0
TIMESTEPS=4096
LAMBDA=1.0
N_eig=32

# Only one job; no indexing logic needed
MODE=${MODES[0]}
MX=${MX_VALUES[0]}
MY=${MY_VALUES[0]}

OUT_DIR="results_qq_ny4096/${MODE}/mx${MX}_my${MY}"
mkdir -p "$OUT_DIR"

# Tell your Python code where to write its psutil memory log:
export MEM_LOG_FILE="${OUT_DIR}/mem_log_${MODE}_mx${MX}_my${MY}.txt"
MEM_PROFILE_FILE="${OUT_DIR}/mem_profile_${MODE}_mx${MX}_my${MY}.txt"


# (Optional) where to tee the combined run‐log:
RUN_LOG="${OUT_DIR}/run.log"

# Run simulation
START=$(date +%s)

python RunSimulation_mx_my.py \
  --N_eig "$N_eig" \
  --nx "$NX" --xmin "$XMIN" --xmax "$XMAX" \
  --ny "$NY" --ymin "$YMIN" --ymax "$YMAX" \
  --mx "$MX" --my "$MY" \
  --x0 "$X0" --y0 "$Y0" \
  --vx0 "$VX0" --vy0 "$VY0" \
  --sigmay "$SIGMAY" \
  --total_time "$TOTAL_TIME" \
  --timesteps "$TIMESTEPS" \
  --lambda_ "$LAMBDA" \
  --mode "$MODE" \
  --base "${MODE}_mx${MX}_my${MY}" \
  --output_dir "$OUT_DIR"
 2>&1 | tee "$RUN_LOG"
 
END=$(date +%s)
echo "Finished at: $(date)"          | tee -a "$RUN_LOG"
echo "Elapsed time: $((END-START))s" | tee -a "$RUN_LOG"

#echo " Finished: mode=$MODE, mx=$MX, my=$MY"
#echo "Saving metadata..."
cat > "${OUT_DIR}/params.txt" << EOF
base=${MODE}_mx${MX}_my${MY}
lambda=$LAMBDA
mode=$MODE
mx=$MX
my=$MY
N_eig=$N_eig
nx=$NX
ny=$NY
output_dir=$OUT_DIR
sigmay=$SIGMAY
timesteps=$TIMESTEPS
total_time=$TOTAL_TIME
vx0=$VX0
vy0=$VY0
x0=$X0
xmax=$XMAX
xmin=$XMIN
y0=$Y0
ymax=$YMAX
ymin=$YMIN
EOF

touch "${OUT_DIR}/done.txt"