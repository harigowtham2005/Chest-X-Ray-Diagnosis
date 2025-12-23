#!/bin/bash

echo "Installing requirements..."
pip install -r requirements.txt

echo "Training baseline model..."
python src/train_baseline.py

echo "Training fusion model..."
python src/train_fusion.py

echo "Starting Streamlit app..."
streamlit run app.py
