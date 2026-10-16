# Handwriting Digit Prediction using CNN

A deep learning web application that recognizes handwritten digits from **0 to 9** using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can upload an image of a handwritten digit and receive a prediction through a web interface.

## Features

- Predict handwritten digits from 0 to 9.
- CNN model built using PyTorch.
- Trained on the MNIST handwritten digit dataset.
- Upload handwritten digit images for prediction.
- Display the predicted digit and confidence score.
- Web interface using HTML, CSS, and JavaScript.
- Python backend powered by FastAPI.

## Technologies Used

- Python
- PyTorch
- CNN (Convolutional Neural Network)
- MNIST Dataset
- FastAPI
- HTML
- CSS
- JavaScript
- Pillow (PIL)

## Project Structure

```text
HandWriting_Digit_Predict/
│
├── HandWriting_Digit_Predict.ipynb
├── main.py
├── mnist_cnn.pth
├── index.html
├── script.js
├── style.css
└── README.md
```

## How It Works

1. The CNN model is trained using the MNIST dataset.
2. The trained model weights are saved in `mnist_cnn.pth`.
3. The FastAPI backend loads the trained model.
4. The user uploads an image of a handwritten digit.
5. The image is preprocessed before being passed to the model.
6. The model predicts the digit and returns the prediction with a confidence score.
7. The result is displayed through the web interface.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/HandWriting_Digit_Predict.git
cd HandWriting_Digit_Predict
```

Replace `YOUR_USERNAME` with your GitHub username.

**2. Install dependencies**

```bash
pip install fastapi uvicorn torch torchvision pillow python-multipart
```

**3. Run the application**

```bash
python -m uvicorn main:app --reload
```

**4. Open the application**

Visit the following URL in your browser:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Model Information

- **Dataset:** MNIST
- **Task:** Handwritten digit classification
- **Classes:** 10 (digits 0–9)
- **Model:** Convolutional Neural Network (CNN)
- **Framework:** PyTorch

## Future Improvements

- Improve prediction performance on real-world handwriting.
- Add a drawing canvas for digit input.
- Display top prediction probabilities.
- Deploy the application online.

## Author

**Budhadev Mondal**

GitHub:https://github.com/buddhadevmondal

## License

This project is available for educational and learning purposes.
