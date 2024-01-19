# Topic: Board Games

## Initial Data Preprocessing Requirements

- Data Collection
- Encoding categorical variables
- Feature Scaling
- Handling Missing Values Strategy

The first part of this project aims to preprocess the data to enable a more robust and accurate analysis. To start with preprocessing, you can follow the steps and tasks mentioned above to prepare the data for your further analysis.

## Usage

### Requirements:
- Ensure you have the following libraries installed: `pandas`, `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.metrics`, `lightgbm`, `xgboost`, `catboost`.
- The dataset file `board_games.csv` should be located in the directory: [BoardGames/src/data](./src).

### Steps:

1. **Import Required Libraries:**
   ```
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
   ```

2. **Load Dataset:**
   ```
   dataset = pd.read_csv('board_games.csv')
   ```

3. **Understanding Data:**
   - Display data types and summary statistics of the dataset:
     ```
     print(dataset.dtypes)
     print(dataset.describe())
     ```

4. **Preprocessing data:**
   - Handle missing values in the columns by filling them with the mean:
     ```
     dataset.fillna(dataset.mean(numeric_only=True), inplace=True)  # or dataset.dropna(inplace=True)
     ```
   - Encoding categorical variables:
    ```
    label_encoders = {}
    for column in dataset.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    dataset[column] = label_encoders[column].fit_transform(dataset[column])
     ```
   - Feature Scaling:
   ```
   scaler = StandardScaler()
   scaled_features = scaler.fit_transform(dataset.drop('users_rated', axis=1))
    ```
### Execution in Jupyter Notebook:

- Run `jupyter notebook` in your terminal
- Ensure that the `board_games.csv` dataset file is located in the directory specified [BoardGames/src/data](./src).

Execute the code within the Jupyter Notebook by running each section sequentially. This process will:
- Load the dataset (Data collection)
- Encoding categorical variables
- Feature Scaling
- Handling Missing Values Strategy
- Implement Algorithms of:
+ LightGBM
+ XGBoost
+ CatBoost
- Show results for each algorithm which will be able to have a comparison between them.
