"""
Virtual Environment Setup Script

This script automates the creation and setup of a Python virtual environment
for the DMSL Project.

Usage:
    python setup_venv.py
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"→ {description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}\n")
        return False


def main():
    """Main setup function."""
    print_header("DMSL PROJECT - Virtual Environment Setup")

    # Get project directory
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}\n")

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("✗ Python 3.8 or higher is required!")
        sys.exit(1)
    print("✓ Python version is compatible\n")

    # Step 1: Create virtual environment
    print_header("Step 1: Creating Virtual Environment")
    venv_path = project_dir / "venv"

    if venv_path.exists():
        response = input("Virtual environment already exists. Recreate? (y/n): ")
        if response.lower() == 'y':
            print("Removing existing virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("Using existing virtual environment\n")

    if not venv_path.exists():
        if not run_command(f'{sys.executable} -m venv venv',
                          "Creating virtual environment"):
            sys.exit(1)

    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_executable = venv_path / "Scripts" / "pip.exe"
    else:  # Linux/Mac
        activate_script = venv_path / "bin" / "activate"
        pip_executable = venv_path / "bin" / "pip"

    # Step 2: Upgrade pip
    print_header("Step 2: Upgrading pip")
    if not run_command(f'"{pip_executable}" install --upgrade pip',
                      "Upgrading pip"):
        print("Warning: Failed to upgrade pip, continuing anyway...\n")

    # Step 3: Install requirements
    print_header("Step 3: Installing Requirements")
    requirements_file = project_dir / "requirements.txt"

    if not requirements_file.exists():
        print("✗ requirements.txt not found!")
        sys.exit(1)

    if not run_command(f'"{pip_executable}" install -r requirements.txt',
                      "Installing dependencies"):
        print("✗ Failed to install dependencies!")
        sys.exit(1)

    # Step 4: Test imports
    print_header("Step 4: Testing Installation")

    test_script = """
import sys
try:
    import numpy
    import scipy
    import torch
    import matplotlib
    import pandas
    import sklearn
    import jupyter
    print("✓ All core packages imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
"""

    if os.name == 'nt':
        python_executable = venv_path / "Scripts" / "python.exe"
    else:
        python_executable = venv_path / "bin" / "python"

    test_result = subprocess.run(
        [str(python_executable), "-c", test_script],
        capture_output=True,
        text=True
    )

    print(test_result.stdout)

    if test_result.returncode != 0:
        print("✗ Package testing failed!")
        print(test_result.stderr)
        sys.exit(1)

    # Final instructions
    print_header("Setup Complete!")

    print("Virtual environment created successfully!")
    print(f"\nLocation: {venv_path}\n")

    print("To activate the virtual environment:\n")

    if os.name == 'nt':  # Windows
        print("  Windows CMD:")
        print(f"    venv\\Scripts\\activate.bat\n")
        print("  Windows PowerShell:")
        print(f"    venv\\Scripts\\Activate.ps1\n")
    else:  # Linux/Mac
        print("  Linux/Mac:")
        print(f"    source venv/bin/activate\n")

    print("To launch Jupyter Notebook:")
    print("  1. Activate the virtual environment (see above)")
    print("  2. Run: jupyter notebook")
    print("  3. Navigate to the notebooks/ folder\n")

    print("Recommended notebooks to start with:")
    print("  • 01_lorenz_system.ipynb - Chaotic Lorenz attractor")
    print("  • 03_van_der_pol.ipynb - Oscillatory Van der Pol")
    print("  • 05_comparative_analysis.ipynb - Cross-system comparison\n")

    print("=" * 70)
    print("Happy coding! 🚀")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
