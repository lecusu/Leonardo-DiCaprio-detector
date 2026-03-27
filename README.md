# Leonardo-DiCaprio-detector
# Leo Detector: Is it Leonardo DiCaprio?

A lightweight web application built with Python and Streamlit that uses a custom-trained PyTorch neural network (ResNet18) to detect whether an uploaded image contains Leonardo DiCaprio.

## Features
* **Simple UI:** Easy-to-use Streamlit interface for uploading images.
* **AI Powered:** Uses a ResNet18 deep learning model fine-tuned for binary classification (Leo vs. Not Leo).
* **Fast Inference:** Optimized to run quickly on standard CPUs.

## Prerequisites 

Make sure you have Python installed on your machine. You can install all the required libraries at once by running this command in your terminal:

```bash
pip install streamlit torch torchvision pillow
```
## How to Run the App

1. Download or clone this repository to your local machine.
2. Open your terminal and navigate to the project folder.
3. Run the following command:
```bash
python -m streamlit run app.py
```
4. This will automatically launch the web application in your default web browser. Just upload an image and let the model do the rest!

## File Structure
* `app.py`: The main Streamlit application code.
* `leo_model.pth`: The saved PyTorch weights for the trained ResNet18 model.
