#!/bin/bash
echo "Running all simulations locally..."

mkdir -p logs_qq_local

# Conda activation
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate py310_env

PARAM_FILE="param_list_qq.txt"
TOTAL_LINES=$(wc -l < "$PARAM_FILE")

for (( i=2; i<=TOTAL_LINES; i++ ))
do
  echo "Running line $i of $TOTAL_LINES"

  LINE=$(sed -n "${i}p" "$PARAM_FILE")

  IFS=',' read -r MODE MX MY X0 VX0 NX XMIN XMAX NY YMIN YMAX Y0 VY0 SIGMAY TOTAL_TIME TIMESTEPS LAMBDA N_EIG FILENAME <<< "$LINE"
  
  OUT_DIR="results_coherent/${MODE}/${FILENAME}"
  mkdir -p "$OUT_DIR"

  export MEM_LOG_FILE="${OUT_DIR}/mem_log.txt"

  RUN_LOG="${OUT_DIR}/run.log"
  START=$(date +%s)

  python RunSimulation_coherent.py \
    --mode "$MODE" --mx "$MX" --my "$MY" \
    --x0 "$X0" --vx0 "$VX0" \
    --nx "$NX" --xmin "$XMIN" --xmax "$XMAX" \
    --ny "$NY" --ymin "$YMIN" --ymax "$YMAX" \
    --y0 "$Y0" --vy0 "$VY0" --sigmay "$SIGMAY" \
    --total_time "$TOTAL_TIME" --timesteps "$TIMESTEPS" \
    --lambda_ "$LAMBDA" --N_eig "$N_EIG" \
    --base "$FILENAME" --output_dir "$OUT_DIR" \
    2>&1 | tee "$RUN_LOG"

  END=$(date +%s)
  echo "Finished at: $(date)"          | tee -a "$RUN_LOG"
  echo "Elapsed time: $((END-START))s" | tee -a "$RUN_LOG"

  cat > "${OUT_DIR}/params.txt" << EOF
base=$FILENAME
mode=$MODE
mx=$MX
my=$MY
x0=$X0
vx0=$VX0
xmin=$XMIN
xmax=$XMAX
y0=$Y0
vy0=$VY0
ymin=$YMIN
ymax=$YMAX
nx=$NX
ny=$NY
sigmay=$SIGMAY
timesteps=$TIMESTEPS
total_time=$TOTAL_TIME
lambda=$LAMBDA
N_eig=$N_EIG
output_dir=$OUT_DIR
EOF

  touch "${OUT_DIR}/done.txt"

  echo "----------------------------------------"
done