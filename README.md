# Crop Recommendation System

## Project Description
This project aims to predict the best crop for a given soil type based on various environmental factors. The system receives data from microcontroller sensors, including temperature, humidity, pH value, and moisture content, and uses this information to recommend the most suitable crop.

## Project Structure
The project is organized into the following folders:

1. **app**:
   - **backend**: This folder contains the FastAPI-based backend, which is deployed on AWS Lambda with a GitHub Actions-based Continuous Deployment (CD) pipeline.
   - **frontend**: The frontend is built using Streamlit and is deployed on Streamlit Cloud.
2. **notebook**: This folder contains Jupyter Notebooks used for data exploration, model development, and testing.
3. **scripts**: This folder holds a script for data preprocessing and model training.
4. **test**: This folder contains unit tests for the backend application.
5. **models**: This folder stores the trained machine learning models used for the crop recommendation.
6. **hardware**: This folder includes scripts that can be used with an Arduino microcontroller to collect sensor data and integrate with the crop recommendation system.
7. **data**: This folder contains the dataset used for training and evaluating the machine learning models.

## Technologies Used
- **Backend**: FastAPI, deployed on AWS
- **Frontend**: Streamlit, deployed on Streamlit Cloud
- **Data Preprocessing and Modeling**: Jupyter Notebooks, Pandas, Scikit-learn
- **Experiment Tracking**: MLflow
- **Hardware Integration**: Arduino

## Project Setup
1. Set up the backend:
   - Clone the repository and navigate to the `app/backend` directory.
   - Install the required dependencies using `pip install -r requirements.txt`.
   - Configure the necessary AWS credentials:
        - Create an AWS IAM user with the appropriate permissions to interact with ECR (Elastic Container Registry) and Lambda.
        - Obtain the access key and secret key for the IAM user.
        - Set these credentials as secrets in your GitHub repository.
   - Setup all lambda functions and ECR repo on AWS console (One time)
        - Create a lambda function on AWS lambdas from ecr images
        - Enable Function url
        - From the settings make allow cors "*" or specify the source that will call the function
2. Set up the frontend:
   - Navigate to the `app/frontend` directory.
   - Install the required dependencies using `pip install -r requirements.txt`.
   - Change the api endpoint to the one you just deployed from aws lambda
   - Deploy the Streamlit-based frontend to Streamlit Cloud.
3. Explore the notebooks, scripts, and test cases in the respective folders.
4. Integrate the hardware (Arduino) with the crop recommendation system using the scripts in the `hardware` folder.

## Experiment Tracking with MLflow
The project uses MLflow to track the various experiments and models. You can access the MLflow UI to view the experiments, compare different models, and select the best-performing one for deployment.

## Contribution
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create a new issue or submit a pull request.