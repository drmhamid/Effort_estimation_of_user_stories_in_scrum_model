# Effort_estimation_of_user_stories_in_scrum_model
A Novel Stacking-Based Ensemble Machine Learning Model for Accurate User Story Effort Estimation in Scrum
This repository contains the source code, dataset, and web-based prototype for the research paper titled "A Novel AI-Based Ensemble Model for Accurate User Story Effort Estimation in Scrum".
The project introduces a stacking-based ensemble model using Extra Trees, XGBoost, and Random Forest as Tier-1 base learners, and Linear Regression as a Tier-2 meta-learner to enhance the accuracy of effort estimation in Agile/Scrum environments.


Repository Structure
The repository is organized as follows:
📂 models: Contains the serialized machine learning models (.pkl files) after training.
📂 static: Includes CSS, JavaScript, and image assets for the web-based interface.
📂 templates: Contains HTML files for the Flask-based web application.
📄 app.py: The main Flask application script that serves the web interface and handles real-time estimations.
📄 train_and_save_model.py: Python script used to preprocess data, train the ensemble model, and save the finalized weights to the models directory.
📊 FINALIZED_DATASET.xlsx: The high-fidelity dataset comprising 160 user stories from 36 professional Scrum-based projects.
🗄️ projects.db: SQLite database file used to store project data and user story inputs.


Key Features
13 Effort Drivers: The model utilizes industry-validated drivers like Complexity, Priority, and Team Experience.
Stacking Ensemble: Mathematically combines multiple regressors to minimize predictive bias and variance.
Web Prototype: A functional interface for Scrum Masters to perform real-time data-driven estimations.



Installation and Setup
Clone the Repository:
git clone https://github.com/drmhamid/Effort_estimation_of_user_stories_in_scrum_model.git
cd Effort_estimation_of_user_stories_in_scrum_model
Install Prerequisites:
Ensure Python 3.x is installed. Install the required libraries via pip:
pip install flask pandas scikit-learn xgboost openpyxl
Train the Model (Optional):
If you wish to retrain the model on the provided dataset:
python train_and_save_model.py
Run the Application:
Launch the web interface locally:
python app.py
Access the tool at http://127.0.0.1:5000 in your web browser.


Dataset Availability
The dataset used in this study is included in the repository as FINALIZED_DATASET.xlsx. It consists of 160 user stories with 13 granular effort drivers determined through systematic review and industry cooperation.



Citation
If you find this research or code useful, please cite our paper:
(Insert your paper's citation here once published)


Contact
For any queries or collaboration requests, please contact the corresponding author at: mhamid@gcwus.edu.pk
