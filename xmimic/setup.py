from setuptools import find_packages, setup

setup(
    name='humanx',
    version='0.0.1',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos',
    url="https://github.com/wyhuai/HumanX",  # Update this with your actual repository URL
    python_requires=">=3.8",
    install_requires=[
        "hydra-core>=1.2.0",
        "numpy==1.23.5",
        "rich",
        "ipdb",
        "matplotlib",
        "termcolor",
        "wandb",
        "plotly",
        "tqdm",
        "loguru",
        "meshcat",
        "pynput",
        "scipy",
        "tensorboard",
        "onnx",
        "onnxruntime",
        "opencv-python",
        "joblib",
        "easydict",
        "lxml",
        "numpy-stl",
        "open3d"
    ]
)