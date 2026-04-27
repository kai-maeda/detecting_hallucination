//Go to OpenAI and setup API key

//Connect to runpod H100 SXM
//Bump container disk to 50GB for QWEN weights + videos

//Connect to the Pod Web Terminal

//To deteatch inference run from SSH
apt-get update && apt-get install -y tmux

//Start a session
tmux new -s exp

cd/workspace
git clone https://github.com/kai-maeda/detecting_hallucination.git
cd detecting_hallucination

bash setup_runpod.sh

//Add OpenAI API key
cp .env.example .env
nano .env

//Sanity check VSI-Bench fields
python3 scripts/01_inspect_dataset.py

//Download videos + sample 50 per category 
python3 scripts/02_prep_data.py

//Generate paraphrases via GPT-4o-mini 
python3 scripts/03_paraphrase.py

// Run Qwen2.5-VL-7B inference 
python3 scripts/04_inference.py

// Score
python3 scripts/05_score.py

// Plot figure + LaTeX table
python3 scripts/06_plot.py

//download figues/ and results/ from runpod

//Stop pod

Things that might bite you (and how to handle)
VSI-Bench field names: my code assumes question_type, ground_truth, dataset, scene_name, options. The inspect script tells you the truth. Adjust the constants at the top of 02_prep_data.py if needed.
Video download paths: 02_prep_data.py tries 4 path patterns; if all fail for everything, the dataset stores videos differently (could be a tarball or symlinked from ScanNet++). You'll see "0 videos downloaded" — ping me with what 01_inspect_dataset.py printed and I'll adjust.
flash-attn install: sometimes fails on weird CUDA versions. The inference script catches this and falls back to SDPA automatically (~20% slower, no big deal).
OOM: shouldn't happen on H100 (80GB) with a 7B model + 8 frames. If on smaller GPU, drop N_VIDEO_FRAMES to 4 in src/config.py.