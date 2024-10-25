import random
import logging
import streamlit as st 
import numpy as np 
import pandas as pd 
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

#Create logs folder if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
# Set up logging
logging.basicConfig(
    filename='logs/model_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Set seed 
seed = random.randint(0,100)
df = pd.read_csv('data/preprocessed_used_cars.csv')
#PREPARE DATA FEATURES 
#get the marks of the cars 
marks = list(df.mark.unique())
#grab models for each car mark 
models_of_marks = dict()
for mark in marks:
    models = list(df[df['mark']==mark]['model'].unique())
    models_of_marks[str(mark)] = models
# filter the wanted fuel types 
fuel_types = list(df.fuel_type.unique())
# get the min/max of model year 
min_myear = int(df.production_year.min())
max_myear = int(df.production_year.max())
list_years = list(range(min_myear,max_myear+1)) + [2019,2020]
list_years = sorted(list_years)
#encode categorical variables 
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
df['fuel_type'] = encoder1.fit_transform(df.fuel_type)
df['mark'] = encoder2.fit_transform(df.mark)
df['model'] = encoder3.fit_transform(df.model)
# BUILD the APP WIDGETS 
st.title('Want to sell your car? or buy a used one?')
st.subheader('Get to know its pricing value before making your decision!')
col1, col2 = st.columns([4,2])
with col1:
    st.image('images/mercedes.jpg',use_column_width=True)
with col2:
    print('\n')
    #insert selectbox for mark 
    mark = st.selectbox('Select Brand',marks)
    # get the mark encoding
    mark_code = int(encoder2.transform([mark]))
    #based on the car mark choose, choose its model 
    switcher = {mark:st.selectbox('Select Model',models_of_marks[mark])}
    car_model = switcher.get(mark)
    #get car model encoding
    car_model_code = int(encoder3.transform([car_model]))
    #insert select box for fuel type
    fuel_type =  st.selectbox('Select Fuel Type',fuel_types)
    # get car fuel type encoding
    fuel_type_code = int(encoder1.transform([fuel_type]))
    #insert slider for model year
    model_year = st.selectbox('Select Production Year',list_years)  
    #insert input number for mileage or slider 
    mileage = st.number_input('Enter Car Mileage')
    #insert input number for for fiscal power 
    fiscal_power = st.number_input('Enter Fiscal Power',value=0)
    #encode variables and scale them beforme making prediction 
    inputs = np.array([model_year,fuel_type_code,mark_code,car_model_code,mileage, fiscal_power]).reshape(1,-1)
        
    if st.button('Predict'):
        # TRAIN THE MODEL 
        X = df.drop(['price'],axis =1)
        y = df.price
        logy = np.log(y)
        #encode categorical variables 
        encoder = LabelEncoder()
        #get the categorical columns 
        cat_cols = list(X.select_dtypes(include=['object']))
        for col in cat_cols:
            try:
                X[col] = encoder.fit_transform(X[col])
            except:
                print('Error encoding '+col) 
        #fit the scaler on training data 
        scaler = StandardScaler()
        scaler.fit(X) 
        #transform 
        X_scaled = scaler.transform(X)
        #split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, logy, test_size = 0.2, random_state = seed)
        gbr = GradientBoostingRegressor(loss ='huber', max_depth=6)
        #fit the model on training data
           # Log training attempt
        logging.info(f'Training {gbr}')
        st.write("Training the Model ...") 
        gbr.fit(X_train, y_train)
        logging.info(f'Training is completed sucessefully!')
        #insert a predict button for the price
        scaled_inputs = scaler.transform(inputs)
        price = gbr.predict(scaled_inputs)
        estimated_price = round(float(np.exp(price))/10, 2)
        # show the results in a table 
        results = pd.DataFrame(data = {"Mark": mark, "Model":car_model,"Production Year":model_year,
                                       "Mileage": mileage, "Fuel Type": fuel_type,
                                       "Fiscal Power": fiscal_power, "Predicted Price": estimated_price}, index=[0])
        
        st.table(results)
        # st.write(f"Car Mark:\t {mark}\n")
        # st.write(f"Car Model:\t {car_model}\n")
        # st.write(f"Production Year:\t {model_year}\n")
        # st.write(f"Car Mileage:\t {mileage}\n")
        # st.write(f"Fuel Type:\t {fuel_type}\n")
        # st.write(f"Fiscal Power:\t {fiscal_power} \n")
        # st.write(f"Estimated Price ($):\t {round(estimated_price,2)}\n")
        #st.write(f"Estimated Price ($):\t {round(estimated_price/10,3)}\n")
        logging.info(f"Results: {dict(results.T.values)}")


