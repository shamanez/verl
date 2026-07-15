# Goal — communication-efficient GRPO

This project studies communication-efficient GRPO with two training circuits.

- **Fast circuit:** pipeline parallelism is required. Transformer layers are
  split into chunks across devices, and communication is compressed at pipeline
  boundaries. The research setup is a controlled proxy for the intended
  setting: many small computers connected over the internet.
- **Slow circuit:** a separate device periodically performs dense computation
  and returns a gradient signal that stabilizes the fast circuit. Cadence,
  delay, and latency model the intermittent and stale updates expected in a
  realistic deployment.

## Starting point

Start with `Qwen/Qwen2.5-Math-1.5B` on MATH train/test and the current
PowerSGD, dense-anchor, RELEX, and signed-EMA settings in `project.yaml`.
Compare against ordinary dense GRPO with the same model, data, and training
settings.

## Objective

First make the two-circuit pipeline stable end to end. Then reach parity with
normal GRPO and, if possible, surpass it while reducing communication across
the fast circuit.
