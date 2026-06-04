#!/usr/bin/env bash
# EXP-23 back-to-back arm driver. Runs A1 -> A2 -> A3 sequentially on the warm
# 4xH200 box (each arm uses the full 4-GPU allocation; cannot fan out 3
# multi-GPU arms across 4 GPUs). ';' (not '&&') so an experiment-failure in one
# arm still runs the rest -- that failure is data we paid for (orchestrator rule).
RUN_DIR=/workspace/runs/EXP-23
echo "START_ARMS $(date -Iseconds)" > "$RUN_DIR/arms_driver.log"
for ARM in A1 A2 A3; do
  echo "=== $(date -Iseconds) launching $ARM ===" >> "$RUN_DIR/arms_driver.log"
  bash "$RUN_DIR/launch.sh" "$ARM" >> "$RUN_DIR/arms_driver.log" 2>&1
  echo "=== $(date -Iseconds) $ARM exited rc=$? ===" >> "$RUN_DIR/arms_driver.log"
done
echo "$(date -Iseconds) all-arms-done" > "$RUN_DIR/done.flag"
echo "=== $(date -Iseconds) ALL ARMS DONE ===" >> "$RUN_DIR/arms_driver.log"
