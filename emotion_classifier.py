import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class EmotionClassifier:
    def __init__(self):
        self.le = LabelEncoder()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'realease_year']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['artist'])
            ])
        self.pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('f_selector', RFECV(estimator=RandomForestClassifier(random_state=42), step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1))
        ])
        self.artist_mood = None
        self.best_rf = None
        self.feature_names = None

    def preprocess_data(self, data):
        # Convert 'danceability' column to float
        data['danceability'] = pd.to_numeric(data['danceability'], errors='coerce')

        # Remove rows with missing values
        data.dropna(inplace=True)

        # Remove unnecessary columns
        data.drop(['id', 'name'], axis=1, inplace=True)

        # Encode emotion_tag into numerical values
        data['emotion_tag'] = self.le.fit_transform(data['emotion_tag'])

        # Create an artist-based model
        self.artist_mood = data.groupby('artist')['emotion_tag'].agg(lambda x: x.value_counts().index[0])

        return data

    def train(self, data):
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(data.drop('emotion_tag', axis=1), data['emotion_tag'], test_size=0.2, random_state=42)
        
        # Store the X_train column names as a class attribute
        self.feature_names = X_train.columns.tolist()

        # Pipeline for preprocessing and feature selection
        X_train_transformed = self.pipe.fit_transform(X_train, y_train)
        X_val_transformed = self.pipe.transform(X_val)

        # Define a Random Forest model with grid search and cross-validation
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_transformed, y_train)
        self.best_rf = grid_search.best_estimator_

        # Evaluate performance on the validation set
        rf_preds = self.best_rf.predict(X_val_transformed)

        print('Random Forest:')
        print('Accuracy:', accuracy_score(y_val, rf_preds))
        print(classification_report(y_val, rf_preds))


    def predict(self, test_data):
        # Preprocess test data
        test_data.drop(['name'], axis=1, inplace=True)
        test_data['danceability'] = pd.to_numeric(test_data['danceability'], errors='coerce')
        
        # Use the stored feature_names attribute
        test_data = test_data[self.feature_names + ['id']]

        # Feature scaling and transformation for the test set
        test_data_transformed = self.pipe.transform(test_data.drop('id', axis=1))

        # Make predictions using the trained Random Forest model
        test_preds = self.best_rf.predict(test_data_transformed)

        # Create a DataFrame with the predicted values and the corresponding IDs
        results = pd.DataFrame({'id': test_data['id'], 'emotion_tag': self.le.inverse_transform(test_preds), 'artist': test_data['artist']})

        # Apply the artist-based model
        results['emotion_tag'] = results.apply(lambda row: self.le.inverse_transform([self.artist_mood[row['artist']]])[0] if row['artist'] in self.artist_mood.index else row['emotion_tag'], axis=1)

        # Save the results to a CSV file
        results[['id', 'emotion_tag']].to_csv('results.csv', index=False)

        return results


if __name__ == '__main__':
    # Load data into pandas DataFrame
    path = './data/val_set.csv'
    data = pd.read_csv(path)
    # Initialize the EmotionClassifier
    emotion_classifier = EmotionClassifier()

    # Preprocess the data
    preprocessed_data = emotion_classifier.preprocess_data(data)

    # Train the classifier
    emotion_classifier.train(preprocessed_data)

    # Load the test data into a pandas DataFrame
    path2 = './data/test_set_no_labels.csv'
    test_data = pd.read_csv(path2)

    # Make predictions and save them to a CSV file
    results = emotion_classifier.predict(test_data)
