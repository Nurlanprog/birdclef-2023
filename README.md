Dataset link: https://www.kaggle.com/datasets/nur373/birdclef-2023.

Insights:
1) CuPy preprocessing beats librosa in speed approximately 5-10x (audio to mel-spectogram conversion).
2) .npy caching > on fly training
3) Multi-gpu training approach with accelerate library achieved on jupyter notebook environment. Problems with notebook_launcher were resolved by saving separate files and running as standalone processes with different CUDA's.
