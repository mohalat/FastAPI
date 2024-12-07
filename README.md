## Sepsis Prediction API
This project implements a Machine Learning (ML) API designed to predict whether a patient in the ICU is at risk of developing sepsis. The API is built with a focus on scalability and performance, leveraging Docker for containerization and ease of deployment.
 
### Table of Contents
-Introduction
-Features
-Dataset Overview
-Attribute Descriptions
-Hypothesis
-Sepsis Target
-Business Questions
-Technologies Used
-Setup Instructions
-Machine Learning Models
-Installation Requirements
-API Endpoints
-Troubleshooting
-Authorship
-Donor Acknowledgement
-License
 
 

 
### Introduction
Sepsis is a life-threatening condition that arises when the body's response to infection causes injury to its tissues and organs. Early prediction of sepsis is crucial in providing timely interventions and improving patient outcomes. This project provides a robust Machine Learning-based API for sepsis prediction using patient medical data.
 
### Features
Predict sepsis risk based on patient medical records.
Scalable and containerized application using Docker.
RESTful API endpoints for integration with healthcare systems or applications.
Accurate predictions powered by pre-trained machine learning models.
 
### Dataset Overview
The dataset used for this project contains medical records of patients admitted to the ICU. The primary goal is to predict the Sepsis target variable, which indicates whether a patient is likely to develop sepsis.

### Column Name Attribute/Target    Description
-ID  N/A Unique number to represent patient ID
-PRG Attribute 1 Plasma glucose
-PL  Attribute 2 Blood Work Result-1 (mu U/ml)
-PR  Attribute 3 Blood Pressure (mm Hg)
-SK  Attribute 4 Blood Work Result-2 (mm)
-TS  Attribute 5 Blood Work Result-3 (mu U/ml)
-M11 Attribute 6 Body Mass Index (BMI): weight in kg/(height in m)^2
-BD2 Attribute 7 Blood Work Result-4 (mu U/ml)
-Age Attribute 8 Patient's age (years)
-Insurance   N/A Indicates if a patient holds a valid insurance card


### Sepsis Target:
Patient in ICU will develop sepsis; Negative: Otherwise
 
### Technologies Used:
-Python: For building the machine learning model and API.
-FastAPI: To develop the RESTful API.
-Scikit-learn: For machine learning model development.
-Docker: For containerization and deployment.
-NumPy and Pandas: For data processing and analysis.
 
### Setup
Sepsis Prediction API
This is a Sepsis Prediction API built using FastAPI. The API allows users to upload a CSV file and predict sepsis risk based on a chosen machine learning model. The results are returned in JSON format, including both the original data and predictions.
 
### Machine Learning Models
Multiple Models: Choose between different pre-trained models for prediction:
 
-"DecisionTree"
-"RandomForest"
-"LogisticRegression"
-"KNN"

CSV File Upload: Upload a CSV file with the necessary features for prediction.
 
Prediction Output: Get a JSON response containing the original data along with the predictions for each row in a new column labeled Prediction.
 
### Installation Requirements
-Python 3.7 or higher
-FastAPI
-Uvicorn
-Pandas
-Scikit-learn
-Matplotlib
-Seaborn
-scipy
 

Step 1: Clone the repository -->
git clone <repository-url>
cd <repository-directory>
 
Step 2: Install dependencies:
Create a virtual environment and install the required dependencies using pip:
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

Step 3: Run the FastAPI server
To start the FastAPI application, run the following command:
uvicorn mlapi:app --reload.
This will start the FastAPI app locally at http://127.0.0.1:8000.
Copy URL and paste it in a new browser and add /docs eg. http://127.0.0.1:8000/docs to  access the API.
 
### API Usage
-Endpoint: /predict
-Method: POST

This endpoint allows you to make predictions using a specified model.
 
Type in the name of the model you want to use for prediction.
 ![alt text](<Screenshot 2024-12-07 at 02.16.51.png>)

-Model – The name of the model you wish to use for prediction. 
-Available models:
"DecisionTree",
"RandomForest",
"LogisticRegression",
"KNN".
 
Upload File:
Navigate to the "Upload File" section and select a CSV file.
![alt text](<Screenshot 2024-12-07 at 02.38.21.png>)
 
file (file) – A CSV file with the necessary features for prediction. The file should contain the following columns:
-"PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age", "Insurance"
 
### Response:
A JSON response containing the predictions for each row in a new column labeled "Prediction".
 ![alt text](<Screenshot 2024-12-07 at 02.17.02.png>)
 
 
### Troubleshooting
1. Incorrect Number of Features
If the number of features in the uploaded CSV file does not match what the model expects, the API will return an error. Ensure that the CSV file has the correct number of columns.
 
2. Model Not Found
If an invalid model is selected, the API will return a 400 error stating that the model was not found.
 
3. Invalid File Format
If the uploaded file is not a CSV file, the API will return a 400 error.
 
### Glance view of API application
 ![alt text](<Screenshot 2024-12-07 at 02.16.51.png>) ![alt text](<Screenshot 2024-12-07 at 02.17.02.png>) ![alt text](<Screenshot 2024-12-07 at 02.17.23.png>) ![alt text](<Screenshot 2024-12-07 at 02.17.30.png>)
 


### About the Author
This FastAPI application was developed by Abdul-Latif Mohammed, a passionate data analyst and machine learning enthusiast with a strong background in data analysis and artificial intelligence. Abdul is dedicated to leveraging machine learning models to deliver actionable insights and accurate predictions.

In this project, Abdul integrated a machine learning model to predict medical conditions, creating an intuitive API that allows users to upload data and receive predictions seamlessly. The primary goal is to simplify the interaction between complex machine learning models and end-users, making advanced technologies more accessible and practical.
 
Abdul is committed to developing scalable, efficient, and impactful machine learning solutions that can be applied across diverse
industries, including healthcare and business analytics.
 
 
### Donor of database:
The Johns Hopkins University
Johns Hopkins Road
Laurel, MD 20707
(301) 953-6231
 
 
 
### License
This project is licensed under the MIT License. See the LICENSE file for details.
has context menu
