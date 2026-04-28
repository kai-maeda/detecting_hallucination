# Detecting Spatial Hallucination — Run Instructions

Step-by-step instructions for reproducing the experiment on RunPod.

## 1. Prerequisites

- OpenAI API key — create at https://platform.openai.com/api-keys
- RunPod account with billing enabled

## 2. Provision the pod

- Spin up a **RunPod H100 SXM** pod
- Bump container disk to **50 GB** (for Qwen weights + videos)
- Connect via the **Pod Web Terminal**

## 3. Set up `tmux` so the inference run survives SSH disconnects

```bash
apt-get update && apt-get install -y tmux
tmux new -s exp
```

## 4. Clone and install

```bash
cd /workspace
git clone https://github.com/kai-maeda/detecting_hallucination.git
cd detecting_hallucination
bash setup_runpod.sh
```

## 5. Add your OpenAI API key

```bash
cp .env.example .env
nano .env   # paste your sk-... key, save with Ctrl+O, Ctrl+X
```

## 6. Run the pipeline

```bash
# Sanity-check VSI-Bench fields
python3 scripts/01_inspect_dataset.py

# Download videos + sample N per category
python3 scripts/02_prep_data.py

# Generate paraphrases via GPT-4o-mini
python3 scripts/03_paraphrase.py

# Run Qwen2.5-VL-7B inference
python3 scripts/04_inference.py

# Score
python3 scripts/05_score.py

# Plot figures + LaTeX results table
python3 scripts/06_plot.py
```

## 7. Pull artifacts back

Download `figures/` and `results/` from the pod (web file browser, JupyterLab, or `scp`).

## 8. Stop the pod

Terminate from the RunPod dashboard to stop billing.
