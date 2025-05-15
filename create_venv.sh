#!/bin/bash

# Script to set up the Python environment for VitalDB Jepa preprocessing

# Define the name of the virtual environment directory
VENV_DIR="./.jepa_env"

# Define Python version (optional, python3 should be sufficient)
PYTHON_CMD="python3"

echo "--- Python Environment Setup for VitalDB Jepa Preprocessing ---"

# Check if python3 is available
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "$PYTHON_CMD could not be found. Please install Python 3."
    exit 1
fi

echo "Found Python 3: $($PYTHON_CMD --version)"

# Check if virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " RECREATE_VENV
    if [[ "$RECREATE_VENV" == "y" || "$RECREATE_VENV" == "Y" ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        echo "Creating a new virtual environment in '$VENV_DIR'..."
        $PYTHON_CMD -m venv $VENV_DIR
    else
        echo "Using existing virtual environment."
    fi
else
    echo "Creating a new virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Check if virtual environment was created successfully
if [ ! -d "$VENV_DIR" ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

echo "Activating virtual environment..."
# Source the activation script. Note: this only activates it for the current script's execution.
# The user will need to run this command manually in their terminal as well.
source "$VENV_DIR/bin/activate"

echo "Virtual environment activated. Python version in venv: $(python --version)"

echo "Installing required Python packages..."
# List of packages to install
PACKAGES=(
    "vitaldb"
    "numpy"
    "scipy"
    "pandas"
    "tqdm"
    "matplotlib"
    "torch"
    "scikit-learn"
)

for PACKAGE in "${PACKAGES[@]}"; do
    echo "Installing $PACKAGE..."
    pip install "$PACKAGE"
    if [ $? -ne 0 ]; then
        echo "Failed to install $PACKAGE. Please check your network connection and try again."
        # Optionally, you might want to exit here or try to continue
        # exit 1
    fi
done

echo "--- Setup Complete ---"
echo "To activate the virtual environment in your current terminal session, run:"
echo "  source $VENV_DIR/bin/activate"
echo "To deactivate it later, simply run:"
echo "  deactivate"

exit 0
