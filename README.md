# Road Surface Classification using CNN

## Project Overview

This project uses a Convolutional Neural Network (CNN) to classify road surface types, capable of identifying the following 6 road types:
- Dirt roads
- Asphalt roads
- Gravel roads
- Muddy roads
- Snow-covered roads
- Brick roads

## Project Structure

```
road-surface-classification/
├── image/                    # Folder for training images
├── labels.json               # Image annotation file
├── Road_recognition_CNN.py   # Main program file
├── road_surface_classifier.h5 # Trained model file
├── loss_curve.png            # Training loss curve graph
└── README.md                 # Project documentation
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Prepare the dataset:
   - Place all images in the `image/` folder
   - Ensure the `labels.json` file contains correct image annotations

2. Run the training script:
```bash
python Road_recognition_CNN.py
```

3. After training completes, the following will be generated:
   - `road_surface_classifier.h5` - Trained model

## Model Architecture

The CNN model consists of the following layers:
1. Input layer (224x224 RGB image)
2. Conv2D layer (32 3x3 filters, ReLU activation)
3. Max pooling layer (2x2)
4. Conv2D layer (64 3x3 filters, ReLU activation)
5. Max pooling layer (2x2)
6. Conv2D layer (128 3x3 filters, ReLU activation)
7. Max pooling layer (2x2)
8. Flatten layer
9. Fully connected layer (6 output nodes, Softmax activation)

## Training Parameters

- Image size: 224x224
- Batch size: 8
- Epochs: 15
- Validation split: 20%
- Optimizer: Adam
- Loss function: Binary cross-entropy

## Visualization

The training process automatically generates a loss curve graph showing the changes in training and validation loss.


## License

MIT License
