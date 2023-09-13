import streamlit as st
import tensorflow as tf
import numpy as np

# Load your trained ANN model
model = tf.keras.models.load_model('D:\kishore.h5')

# Define a function to make predictions
def predict_yield(id, clone_size, honeybee, bumbles, andrena, osmia, max_upper_t_range, min_upper_t_range, avg_upper_t_range, max_lower_t_range, min_lower_t_range, avg_lower_t_range, raining_days, avg_raining_days, fruit_set, fruit_mass, seeds, crop_yield):
    # Function logic here
    # Perform inference with the loaded model
    input_features = [id, clonesize, honeybee, bumbles, andrena, osmia, MaxOfUpperTRange, MinOfUpperTRange, AverageOfUpperTRange, MaxOfLowerTRange, MinOfLowerTRange, AverageOfLowerTRange, RainingDays, AverageRainingDays, fruitset, fruitmass, seeds, crop_yield]
    prediction = model.predict(np.array([input_features]))

    # You can post-process the prediction as needed
    return prediction[0][0]

# Streamlit UI
st.title('Blueberry Yield Prediction')

# Input fields for user to enter data
id = st.number_input('id:')
clonesize = st.number_input('clonesize:')
honeybee = st.number_input('honeybee:')
bumbles = st.number_input('bumbles:')
andrena = st.number_input('andrena:')
osmia = st.number_input('osmia:')
MaxOfUpperTRange = st.number_input('MaxOfUpperTRange:')
MinOfUpperTRange = st.number_input('MinOfUpperTRange:')
AverageOfUpperTRange = st.number_input('AverageOfUpperTRange:')
MaxOfLowerTRange = st.number_input('MaxOfLowerTRange:')
MinOfLowerTRange = st.number_input('MinOfLowerTRange:')
AverageOfLowerTRange = st.number_input('AverageOfLowerTRange:')
RainingDays = st.number_input('RainingDays:')
AverageRainingDays = st.number_input('AverageRainingDays:')
fruitset = st.number_input('fruitset:')
fruitmass = st.number_input('fruitmass:')
seeds	 = st.number_input('seeds	:')
crop_yield = st.number_input('Crop Yield:')


if st.button('Predict Yield'):
    # Make a prediction when the button is clicked
    predicted_yield = predict_yield(tid, clonesize, honeybee, bumbles, andrena, osmia, MaxOfUpperTRange, MinOfUpperTRange, AverageOfUpperTRange, MaxOfLowerTRange, MinOfLowerTRange, AverageOfLowerTRange, RainingDays, AverageRainingDays, fruitset, fruitmass, seeds, crop_yield)
    st.write(f'Predicted Yield: {predicted_yield:.2f} kg')