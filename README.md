# AI Job Market Salary Prediction

A machine learning project that predicts salary ranges for AI/ML job postings using a Multi-Layer Perceptron (MLP) neural network. The model analyzes job features including skills, tools, location, employment type, job title, industry, and company to predict salary categories.

## Project Overview

This project implements a neural network solution to predict salary ranges for AI job market positions. The model uses PyTorch to train an MLP classifier that takes encoded job features as input and predicts salary categories. A demo GUI application is included to demonstrate the trained model in action.

## Tech Stack

- **Python 3.10+**
- **PyTorch** - Neural network framework
- **Pandas** - Data manipulation and preprocessing
- **NumPy** - Numerical computations
- **Joblib** - Model serialization
- **Pickle** - Encoding mappings serialization
- **Jupyter Notebook** - Development environment

## Dataset

The project uses `ai_job_market.csv` containing 2,002 AI/ML job postings with features including:
- Skills required
- Tools preferred
- Location
- Employment type
- Job title
- Industry
- Company name
- Salary ranges (target variable)

## File Structure

```
ai-job-analysis-model/
├── AIJobMarket.ipynb          # Main Jupyter notebook with training code
├── ai_job_market.csv           # Dataset (not included in repo)
├── model_state.pth             # Saved PyTorch model state dict
├── encoding_mappings.pkl       # Saved encoding mappings for deployment
├── demo_app.py                 # Demo GUI application for salary prediction
├── mlp_model.joblib            # Legacy model file (not used)
├── my.py                       # Utility functions
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## How to Run

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

### Training the Model

1. Open `AIJobMarket.ipynb` in Jupyter Notebook or VS Code
2. Ensure `ai_job_market.csv` is in the same directory
3. Run all cells sequentially

The notebook includes:
- Data preprocessing and feature encoding
- Train/test split (80/20)
- Model training and evaluation
- Model saving (`model_state.pth` and `encoding_mappings.pkl`)

### Running the Demo Application

1. Ensure the model files (`model_state.pth` and `encoding_mappings.pkl`) exist (created by running the notebook)
2. Run the demo application:
```bash
python demo_app.py
```

The demo application provides a GUI interface to predict salary ranges for AI/ML job postings. Enter job details including title, industry, employment type, experience level, company size, and optionally skills and tools. Note: The model performs best when skills and tools are provided.

## Model Architecture

Final model: **ImprovedMLPClassifier**
- Input: 12 features
- Hidden layer: 64 neurons with ReLU and Dropout (0.2)
- Output: 20 salary categories

