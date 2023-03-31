# Music Emotion Classifier

This project aims to build a music emotion classifier using a Random Forest model. The classifier takes various features of a song, such as danceability, energy, key, loudness... and then predicts its emotion (happy, sad, angry...)

## How to run the project

1. Download your own dataset and place it in the 'data' directory, or use the data provided in the data folder. The dataset should contain two files: 'val_set.csv' and 'test_set_no_labels.csv'.

2. Create a Python environment and install the required packages:

```sh
python -m venv YOUR_ENV_NAME
source YOUR_ENV_NAME/bin/activate  # (Linux/macOS) or YOUR_ENV_NAME\Scripts\activate (Windows)
pip install pandas scikit-learn numpy
```

3. Run the 'emotion_classifier.py' script:

```sh
python emotion_classifier.py
```

The script will preprocess the data, train the Random Forest model, evaluate its performance on a validation set, and make predictions on the test dataset. The results will be saved to a CSV file named 'results.csv'.

## Code Explanation

The code is organized into a Python class, `EmotionClassifier`, that includes methods for preprocessing the data, training the model, and making predictions. Below is a brief description of each method:

- `preprocess_data()`: This method takes a DataFrame as input and preprocesses the data by converting the 'danceability' column to float, removing rows with missing values, and encoding the 'emotion_tag' column into numerical values. It also creates an artist-based model for later use.

- `train()`: This method trains the Random Forest model using grid search and cross-validation. It splits the data into training and validation sets, preprocesses the data using a pipeline, and evaluates the model's performance on the validation set.

- `predict()`: This method takes a test dataset as input, preprocesses the data, and makes predictions using the trained Random Forest model. It also applies the artist-based model to improve the results. The predictions are then saved to a CSV file.

To run the code, an instance of the `EmotionClassifier` class is created and its methods are called sequentially to preprocess the data, train the model, and make predictions on the test dataset.
