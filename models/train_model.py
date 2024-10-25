import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFE
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Logger setup (if needed)
logging.basicConfig(
    filename='logs/model_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
features = ['production_year', 'mileage', 'fuel_type', 'mark', 'model']
df = pd.read_csv("data/preprocessed_used_cars.csv")
X = df.drop(['price', 'fiscal_power'], axis = 1)
y = df.price
encoder = LabelEncoder()
scaler = StandardScaler()
# Encode categorical features 
cat_cols = list(X.select_dtypes(include=['object']))
for col in cat_cols:
    try:
        X[col] = encoder.fit_transform(X[col])
    except:
        print('Failed Encoding'+col) 
# Scale Data  
X_scaled = scaler.fit_transform(X)
# split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 101)
    # Train the model 
# Function to trigger model training
def train_model():
    # Log start of training
    logging.info('Starting model training...')
    model = GBR(loss ='huber', max_depth=6)
    model.fit(X_train,y_train)
    # Log end of training
    logging.info('Model training completed successfully.')
    return model

# Function to evaluate the performance of the model 
def evaluate_mdoel():
    model = train_model()
    predicted = model.predict(X_test) 
    r2 = r2_score(y_test, predicted)
    logging.info(f' Model Perfoamnce: RMSE={0}, R-squared Score{r2}')
    return r2

#  Function to precit the pricing value of the car 
def predict_price(features):
    # Log start of prediction
    logging.info(f'Price prediction for car features: {features}')
    # Train model 
    model = train_model()
    # Prepare features for prediction
    mark_encoder = LabelEncoder()
    model_encoder = LabelEncoder()
    fuel_encoder = LabelEncoder()
    year = features[0]
    mileage = features[1]
    encoded_fuel = model_encoder.fit_transform([features[2]])
    encoded_mark = mark_encoder.fit_transform([features[3]])
    encoded_model = model_encoder.fit_transform([features[4]])
    feature_vector = np.array([year, mileage, encoded_fuel, encoded_mark, encoded_model]).reshape(1,-1)
    # Use the trained model to predict the car price 
    predicted_price = model.predict(feature_vector) 
    # Log prediction result
    logging.info(f'Predicted price: ${predicted_price}')
    return predicted_price

#test 
if __name__=="__main__":
    model = train_model()
    r2 = evaluate_mdoel()
    price = predict_price([2010,130000,"Diesel","Toyota", "Corolla"])
    print(price[0])
