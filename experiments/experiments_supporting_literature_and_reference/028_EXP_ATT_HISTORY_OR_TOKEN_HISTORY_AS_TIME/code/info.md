# Navigate to the experiment folder
cd "c:\Git\AKIRA\AKIRA\experiments\experiments_supporting_literature_and_reference\028_EXP_ATT_HISTORY_OR_TOKEN_HISTORY_AS_TIME\code"

# Create a new uv virtual environment
uv venv

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install tqdm numpy matplotlib

# Run the experiment
python exp_a_token_domain.py


uv pip install torch torchvision torchaudio
CPU FRIENDLY
