import subprocess
import sys
import os

venv_dir = "venv"

print("Creating virtual environment...")
subprocess.run([sys.executable, "-m", "venv", venv_dir])

# path to pip inside the venv, OS-aware
pip_path = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "pip")

print("Installing requirements...")
subprocess.run([pip_path, "install", "-r", "requirements.txt"])

print("Registering Jupyter kernel...")
subprocess.run([pip_path, "install", "ipykernel"])
python_path = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "python")
subprocess.run([python_path, "-m", "ipykernel", "install", "--user", "--name=group03"])

print("Setup complete!")