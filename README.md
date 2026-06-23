# SMS Spam Detection with LSTM

A text classification project for detecting spam SMS messages using an LSTM-based neural network built with TensorFlow/Keras.

The model classifies SMS messages as either:

* `Ham` — normal message
* `Spam` — unwanted or promotional message

## Overview

This project uses natural language processing and deep learning to classify SMS messages.
The input text is cleaned, tokenized, padded to a fixed length, and passed through an LSTM neural network.

The model uses:

* Text cleaning and preprocessing
* Tokenization
* Sequence padding
* Class weighting for imbalanced data
* Embedding layer
* LSTM layer
* Dropout regularization
* Binary classification with sigmoid output
* Early stopping
* Confusion matrix and training curve visualization
* Manual SMS prediction examples

## Dataset

This project uses the SMS Spam Collection Dataset.

The expected dataset file name is:

```text
spam.csv
```

The script expects `spam.csv` to be placed in the same directory as the Python file.

Dataset sources:

```text
Kaggle:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
```

The dataset contains SMS messages labeled as `ham` or `spam`.

The commonly used `spam.csv` version contains two main columns:

```text
v1  -> label
v2  -> message text
```

The script keeps only these two columns and renames them to:

```text
label
message
```

## Project Structure

```text
sms-spam-detection-lstm
|
|-- task.py
|-- spam.csv
|-- outputs
|   |-- confusion_matrix.png
|   |-- training_validation_accuracy.png
|   |-- training_validation_loss.png
|-- README.md
```

## Requirements

Python version used:

```text
Python 3.11
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## How to Run

Place `spam.csv` next to the Python file.

Then run:

```bash
python task.py
```

The script will:

1. Load the dataset.
2. Clean the SMS messages.
3. Convert labels into numeric values.
4. Split the data into training, validation, and test sets.
5. Tokenize and pad the text messages.
6. Train an LSTM model.
7. Evaluate the model on the test set.
8. Print the classification report.
9. Save evaluation charts in the `outputs` folder.
10. Test the model on several custom SMS examples.

## Text Preprocessing

The preprocessing step includes:

* Converting text to lowercase
* Replacing URLs with the token `link`
* Removing unnecessary characters
* Keeping only letters, numbers, and spaces
* Removing extra spaces

This helps make the input text more consistent before tokenization.

## Model Architecture

The model architecture is:

```text
Embedding
SpatialDropout1D
LSTM
Dense
Dropout
Dense with sigmoid activation
```

Main settings:

```text
Max vocabulary size: 10000
Max sequence length: 40
Embedding dimension: 64
Batch size: 32
Epochs: 8
```

The final sigmoid output represents the probability of the message being spam.

## Training

The dataset is split as follows:

```text
70% training
15% validation
15% testing
```

Class weights are used because the dataset is imbalanced and normal messages are more frequent than spam messages.

Early stopping is also used to reduce overfitting. If validation loss does not improve for several epochs, training stops and the best weights are restored.

## Outputs

The program saves the following files inside the `outputs` folder:

```text
confusion_matrix.png
training_validation_accuracy.png
training_validation_loss.png
```

These images can be used in a report, presentation, or GitHub README.

## Manual SMS Testing

At the end of the script, several example messages are tested manually.

For each message, the program prints:

* input message
* predicted label
* spam probability
* ham probability
* confidence level

Example output format:

```text
Message: Congratulations! You have won a free iPhone.
Prediction: Spam
Spam probability: 0.9821
Ham probability: 0.0179
Confidence: High
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

## Notes

* The dataset file `spam.csv` is not included by default unless explicitly added.
* If you publish this project on GitHub, it is better to include a dataset download link instead of uploading large or externally licensed datasets.
* The `outputs` folder is generated automatically when the script runs.
* This project is intended for educational and portfolio purposes.

## Possible Improvements

Future improvements could include:

* Saving and loading the trained model
* Adding a simple GUI or web interface
* Comparing LSTM with traditional ML models such as Naive Bayes, SVM, and Logistic Regression
* Adding precision-recall and ROC curves
* Improving preprocessing with stemming or lemmatization
* Using pretrained word embeddings
* Adding a separate inference script for custom user input

## License

This project is provided for educational purposes.
Before redistributing the dataset, check the license and usage terms of the original dataset source.
