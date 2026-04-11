#!/bin/bash

# Navigate to the directory where your script is located (optional)
# cd /path/to/your/project

echo "Starting the first script..."
python forward_fem_piola_sample.py 

# The second script will only run after the first one completes
echo "First script finished. Starting the second script..."
python forward_fem_piola_traction_sample.py

echo "All tasks completed successfully."