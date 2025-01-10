# veRL: Quick Installation Guide

## Prerequisites
- NVIDIA GPU with CUDA support
- Docker installed
- Git

## Installation Steps

### 1. Set up Local Repository
```bash
# Create and enter working directory
cd /home/ubuntu/shamane

# Clone both repositories
git clone https://github.com/volcengine/verl
git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git

# Apply Megatron patch
cp verl/patches/megatron_v4.patch Megatron-LM/
cd Megatron-LM
git apply megatron_v4.patch
```

### 2. Launch Docker Container
```bash
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
    -v /home/ubuntu/shamane/verl:/workspace/verl \
    -v /home/ubuntu/shamane/Megatron-LM:/workspace/Megatron-LM \
    verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
```

### 3. Install Packages (Inside Container)
```bash
# Install veRL
cd /workspace/verl
pip3 install -e .

# Install Megatron-LM
cd /workspace/Megatron-LM
pip3 install -e .
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM

# Return to veRL directory for running examples
cd /workspace/verl
```

You should now be in `/workspace/verl` directory, ready to run examples and follow the documentation.

## Next Steps
- Try the [Quickstart Guide](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- Run the [GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)
- Explore [PPO Examples](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

## Note
- The mounted directories persist between container restarts
- You'll need to run the installation steps (step 3) each time you start a new container
- All example paths assume you're working from `/workspace/verl`