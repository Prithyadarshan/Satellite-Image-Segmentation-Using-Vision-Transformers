"""Print setup commands for Colab or a local Python environment."""
PACKAGES = ["torch", "torchvision", "transformers", "accelerate", "albumentations", "opencv-python-headless", "pillow", "matplotlib", "tqdm"]
if __name__ == "__main__":
    print("Install dependencies with:\n  pip install " + " ".join(PACKAGES))
