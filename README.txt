
# Project Setup and Instructions

This document provides step-by-step instructions to set up and run the program, along with the required dependencies.

## Prerequisites
- **Python (version 3.8 or higher)**: Ensure Python is installed. You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).

## Required Python Packages
To run this project, the following Python packages are required:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

## Installation of Packages
Install the required packages using pip and the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running the Program
1. Clone or download the project repository:
   ```bash
   git clone <repository_url>
   ```
   Or download and extract the ZIP file.

2. Navigate to the project directory:
   ```bash
   cd /path/to/project
   ```

3. If running on a Jupyter Notebook:
   Start Jupyter Notebook and open the project file:
   ```bash
   jupyter notebook
   ```

4. Follow the notebook cells in sequence to execute the program.

## Dataset Requirement
Make sure the dataset (e.g., `heart.csv`) is available in the same directory or provide the full path when prompted.

## Notes
- If any issues occur, ensure all required packages are installed correctly.
- For GPU acceleration using TensorFlow, install the appropriate GPU drivers and CUDA. Refer to [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu).
