
#  MNIST Handwritten Digit Classifier from Scratch (NumPy Only)

This project implements a basic **Neural Network** from scratch using only **NumPy**, to classify handwritten digits from the **MNIST** dataset. No external machine learning libraries (like TensorFlow, Keras, or PyTorch) are used — making it ideal for learning how neural networks work under the hood.

---

##  Dataset

The dataset used is the [MNIST handwritten digit dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), specifically the file:

```
mnist_train_small.csv
```

Ensure this file is in your working directory (e.g., `/content/sample_data/` for Google Colab).

---

##  Project Overview

This project:

- Trains a **2-layer neural network** to classify digits (0-9).
- Uses **ReLU activation** in the hidden layer.
- Uses **Softmax activation** in the output layer.
- Performs **gradient descent** for optimization.
- Trains on 59,000 samples and tests on 1,000 samples.

---

##  Network Architecture

- **Input Layer:** 784 nodes (28×28 pixels per image)
- **Hidden Layer:** 10 nodes (ReLU activation)
- **Output Layer:** 10 nodes (Softmax activation for multi-class classification)

### Forward Propagation Flow:

```
Z1 = W1.X + b1  
A1 = ReLU(Z1)  
Z2 = W2.A1 + b2  
A2 = Softmax(Z2)
```

---

##  Dependencies

Install missing packages using:

```bash
pip install numpy pandas matplotlib
```

Required libraries:

- `numpy`
- `pandas`
- `matplotlib`

---

##  Training

Training is done using the following call:

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=500)
```

- Trains for 500 iterations.
- Accuracy improves steadily and can exceed **84%**.

---

##  Testing & Prediction

You can visualize and test predictions using:

```python
test_prediction(index, W1, b1, W2, b2)
```

This displays:

- The actual digit image
- The predicted vs. actual label

---

##  Sample Output (Accuracy Log)

```
Iteration:   0  Accuracy: ~7.8%
...
Iteration: 250  Accuracy: ~79.3%
...
Iteration: 490  Accuracy: ~84.6%
```

---

##  Accuracy Plot

You can visualize model improvement using:

```python
plt.plot(accuracy_history)
plt.title("Accuracy over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
```

---

##  To-Do / Improvements

- Add support for testing on `mnist_test.csv`
- Save/load trained weights for later use
- Implement mini-batch gradient descent
- Add training loss and accuracy plots

---

##  Contributing

Pull requests are welcome. If you have suggestions for improvement, feel free to open an issue or submit a PR.

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

---

##  Acknowledgments

- [Kaggle MNIST CSV Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- The Python & ML open-source community 💙
