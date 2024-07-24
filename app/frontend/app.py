import streamlit as st
import requests
import json

def call_api():
    # Get the input values from the text fields
    input1 = st.session_state.input1
    input2 = st.session_state.input2
    input3 = st.session_state.input3
    input4 = st.session_state.input4
    input5 = st.session_state.input5
    input6 = st.session_state.input6

    payload = {
        "nitrogen": int(input1),
        "potassium": int(input2),
        "temprature": int(input3),
        "humidity": int(input4),
        "ph": int(input5),
        "rainfall": int(input6)
    }

    headers = {
        "Content-Type": "application/json"
    }

    
    response = requests.post('https://y3zh5wguignc2yon6bnxqmhfey0sdumg.lambda-url.us-east-1.on.aws/prediction', headers=headers, data=json.dumps(payload))
    data = response.json()

    # Display the result
    st.write(f"Result: {data['category']}")

# Set the layout of the app
st.title("Crop Recomedation System")

# Create the text input fields with placeholders
st.session_state.input1 = st.text_input("Nitrogen", placeholder="Enter Nitrogen Value")
st.session_state.input2 = st.text_input("Potassium", placeholder="Enter Potassium Value")
st.session_state.input3 = st.text_input("Temprature", placeholder="Enter Temprature Value")
st.session_state.input4 = st.text_input("Humidity", placeholder="Enter Humidity Value")
st.session_state.input5 = st.text_input("PH", placeholder="Enter PH Value")
st.session_state.input6 = st.text_input("Rainfall", placeholder="Enter Rainfall Value")

# Create the button to call the API
if st.button("Crop Recommendation"):
    call_api()