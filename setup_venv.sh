#!/bin/bash

echo "========================================"
echo "EstAlan Development Environment Setup"
echo "========================================"
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "Checking uv installation..."
uv --version
echo

# Create virtual environment
echo "Creating virtual environment..."
uv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -e src
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To activate the virtual environment in the future:"
echo "  source .venv/bin/activate"
echo
echo "To deactivate:"
echo "  deactivate"
echo
