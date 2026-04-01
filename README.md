conda create -n dog_verification python=3.10 -y
conda activate dog_verification
<!-- conda install -c nvidia/label/cuda-12.1.0 cuda-nvrtc cuda-cudart cuda-nvtx -y -->
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install "ffmpeg" -c conda-forge

<!-- uv pip install  git+https://github.c
om/speechbrain/speechbrain.git
pip install -r requirements.txt 
conda install -c conda-forge cudatoolkit=12.1 -->
export HF_TOKEN=YOUR_TOKEN