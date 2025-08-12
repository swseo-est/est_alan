#!/bin/bash

echo "========================================"
echo "Cleaning EstAlan Virtual Environment"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Nothing to clean."
    exit 0
fi

echo "WARNING: This will remove the entire virtual environment!"
echo "All installed packages will be deleted."
echo
read -p "Are you sure you want to continue? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo
echo "Removing virtual environment..."
rm -rf .venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to remove virtual environment"
    exit 1
fi

echo "Virtual environment removed successfully!"
echo
echo "To recreate the environment, run: ./setup_venv.sh"
echo
