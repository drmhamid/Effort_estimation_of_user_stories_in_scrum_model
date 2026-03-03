# app.py (Updated with XGBoost Workaround in /predict)
import flask
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import sqlite3 # SQLite ke liye import

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Change for production

# --- Configuration ---
MODEL_DIR = "models"
DATABASE = 'projects.db'
# --- Filenames to Load (Match train_and_save_model.py) ---
ET_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "et_pipeline_final.joblib")
RF_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "rf_pipeline_final.joblib")
LR_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "lr_pipeline_final.joblib")
XGB_PIPELINE_FILENAME = os.path.join(MODEL_DIR, "xgb_pipeline_final.joblib")

# --- Fibonacci Points & Helper (No Change) ---
FIBONACCI_POINTS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
print(f"Using Fibonacci Points for rounding: {FIBONACCI_POINTS}")
def find_closest_fibonacci(predicted_value):
    if predicted_value is None: return None
    differences = [abs(fib - predicted_value) for fib in FIBONACCI_POINTS]
    min_diff_index = np.argmin(differences)
    return FIBONACCI_POINTS[min_diff_index]

# --- Database Helper Functions (No Changes) ---
def get_db():
    db = getattr(flask.g, '_database', None)
    if db is None:
        db = flask.g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(flask.g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    try:
        with app.app_context():
            db = get_db()
            cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_projects'")
            if not cursor.fetchone():
                print("Creating user_projects table...")
                db.execute('''
                CREATE TABLE user_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id_text TEXT NOT NULL,
                    title TEXT NOT NULL, stories TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
                ''')
                db.commit(); print("Table 'user_projects' created.")
            else: print("Table 'user_projects' already exists.")
    except Exception as e: print(f"Database initialization error: {e}")

init_db()
# --- End Database Setup ---

# --- Load FINAL Models ---
et_pipeline_loaded = None
rf_pipeline_loaded = None
lr_pipeline_loaded = None
xgb_pipeline_loaded = None # Load the full pipeline object

# --- Define the EXACT 11 features and their expected order ---
EXPECTED_MODEL_FEATURES_ORDER = [
    'Size', 'Complexity', 'Priority', 'Noftasks', 'externalhardware',
    'Requirement Volatility', 'Teammembers', 'developmenttype',
    'relatedtechnologies', 'dbms', 'PL'
]
print(f"Model expects features in this order: {EXPECTED_MODEL_FEATURES_ORDER}")
models_loaded = False # Initialize flag
try:
    print(f"\nLoading final ensemble models from '{MODEL_DIR}'...")
    et_pipeline_loaded = joblib.load(ET_PIPELINE_FILENAME)
    print(f" - Loaded: {ET_PIPELINE_FILENAME}")
    rf_pipeline_loaded = joblib.load(RF_PIPELINE_FILENAME)
    print(f" - Loaded: {RF_PIPELINE_FILENAME}")
    lr_pipeline_loaded = joblib.load(LR_PIPELINE_FILENAME)
    print(f" - Loaded: {LR_PIPELINE_FILENAME}")
    xgb_pipeline_loaded = joblib.load(XGB_PIPELINE_FILENAME) # Load the full pipeline
    print(f" - Loaded: {XGB_PIPELINE_FILENAME}")
    print("All final models loaded successfully.")
    models_loaded = True

except FileNotFoundError as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"FATAL Error: Model file not found: {e}")
    print(f"Please run 'python train_and_save_model.py' first.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
except Exception as e:
    print(f"FATAL Error during model loading: {e}")
    print(traceback.format_exc())
# --- End Model Loading ---

# --- HTML Page Routes (No Changes Needed Here) ---
@app.route('/')
def index_page(): return render_template('index.html')

@app.route('/effort_form')
def effort_form_page():
    projects_list = []
    try:
        db = get_db()
        projects_list = db.execute('SELECT id, project_id_text, title FROM user_projects ORDER BY id DESC').fetchall()
    except Exception as e: print(f"Error fetching projects: {e}"); flash('Error loading project list.', 'error')
    return render_template('home.html', projects=projects_list)

# ... (other routes: login, scrum_planning, user_story, get_stories, save_data - no changes) ...
@app.route('/login')
def login_page(): return render_template('login.html')

@app.route('/scrum_planning')
def scrum_planning_page(): return render_template('scrum.html')

@app.route('/user_story')
def user_story_page(): return render_template('form1.html')

@app.route('/api/get_stories/<int:project_db_id>')
def get_project_stories(project_db_id):
    stories_list = []
    try:
        db = get_db()
        project = db.execute('SELECT stories FROM user_projects WHERE id = ?', (project_db_id,)).fetchone()
        if project and project['stories']:
            stories_list = [story.strip() for story in project['stories'].splitlines() if story.strip()]
    except Exception as e: print(f"Error fetching stories for project {project_db_id}: {e}"); return jsonify({'error': f'Database error: {str(e)}'}), 500
    return jsonify({'stories': stories_list})

@app.route('/save_data', methods=['POST'])
def save_project_data():
    if request.method == 'POST':
        project_id = request.form.get('ProjectId')
        project_title = request.form.get('ProjectTitle')
        user_stories = request.form.get('UserStories')
        if not project_id or not project_title or not user_stories:
            flash('Error: All fields are required!', 'error'); return redirect(url_for('user_story_page'))
        try:
            db = get_db(); db.execute('INSERT INTO user_projects (project_id_text, title, stories) VALUES (?, ?, ?)', (project_id, project_title, user_stories)); db.commit()
            flash('Success! Project details saved.', 'success'); print(f"Saved project: ID={project_id}")
        except Exception as e: db.rollback(); print(f"DB Error: {e}"); flash(f'Database Error: {e}', 'error')
        finally: return redirect(url_for('user_story_page'))
    return redirect(url_for('user_story_page'))


# --- API Route for Prediction (Applying XGBoost Workaround) --- ### MODIFIED ###
@app.route('/predict', methods=['POST'])
def predict():
    """Takes JSON data (11 features), predicts using the loaded ensemble (ET, RF, LR, XGB),
       applying workaround for XGBoost, and returns Fibonacci mapped effort."""
    print("\n--- Prediction Request Received ---")

    if not models_loaded:
         print("Prediction failed: Models were not loaded successfully at startup.")
         return jsonify({'status': 'error', 'message': 'Prediction models are not available.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data received'}), 400
        print(f"Received Raw Data: {data}")

        # --- Input Data Parsing, Mapping, Validation for 11 features (No Change) ---
        input_data = {}
        missing_fields = []
        error_messages = []
        field_mappings = {
            'taskSize': {'model_name': 'Size', 'type': int},
            'taskComplexity': {'model_name': 'Complexity', 'type': int},
            'priority': {'model_name': 'Priority', 'type': int},
            'numOfTasks': {'model_name': 'Noftasks', 'type': int},
            'developmentTeam': {'model_name': 'developmenttype', 'type': str},
            'externalHardware': {'model_name': 'externalhardware', 'type': 'binary'},
            'relatedTechnologies': {'model_name': 'relatedtechnologies', 'type': str},
            'databaseSystem': {'model_name': 'dbms', 'type': str},
            'Requirement Volatility': {'model_name': 'Requirement Volatility', 'type': int},
            'teamMembers': {'model_name': 'Teammembers', 'type': int},
            'programmingLanguage': {'model_name': 'PL', 'type': str}
        }
        for form_field, details in field_mappings.items():
            value = data.get(form_field)
            model_feature_name = details['model_name']
            if value is None or str(value).strip() == '': missing_fields.append(form_field); continue
            try:
                if details['type'] == int: input_data[model_feature_name] = int(value)
                elif details['type'] == str: input_data[model_feature_name] = str(value)
                elif details['type'] == 'binary': input_data[model_feature_name] = 1 if str(value).strip().lower() == 'yes' else 0
            except ValueError: error_messages.append(f"Invalid value '{value}' for '{form_field}'.")
            except Exception as e: error_messages.append(f"Error processing '{form_field}': {e}")

        if missing_fields: error_messages.append(f"Missing fields: {', '.join(missing_fields)}")
        if error_messages:
            full_error_message = " | ".join(error_messages)
            print(f"Data Validation Error: {full_error_message}")
            return jsonify({'status': 'error', 'message': full_error_message}), 400

        # --- DataFrame Creation (No Change) ---
        try:
            input_df = pd.DataFrame([input_data], columns=EXPECTED_MODEL_FEATURES_ORDER)
            print(f"DataFrame for prediction (11 features):\n{input_df.to_string()}")
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return jsonify({'status': 'error', 'message': 'Server error creating prediction input.'}), 500

        # --- Model Predictions (Applying XGBoost Workaround) --- ### MODIFIED ###
        predictions_list = []
        model_predictions = {}
        try:
            print("Predicting with loaded ET...")
            model_predictions['et'] = et_pipeline_loaded.predict(input_df)[0]
            predictions_list.append(model_predictions['et'])

            print("Predicting with loaded RF...")
            model_predictions['rf'] = rf_pipeline_loaded.predict(input_df)[0]
            predictions_list.append(model_predictions['rf'])

            print("Predicting with loaded LR...")
            model_predictions['lr'] = lr_pipeline_loaded.predict(input_df)[0]
            predictions_list.append(model_predictions['lr'])

            # --- XGBoost Manual Prediction Workaround ---
            print("Predicting with loaded XGB (manual workaround)...")
            # 1. Get the fitted preprocessor from the loaded xgb_pipeline
            fitted_preprocessor = xgb_pipeline_loaded.named_steps['preprocessor']
             # 2. Transform the input DataFrame
            input_df_transformed = fitted_preprocessor.transform(input_df)
            # 3. Get the fitted XGBoost model from the loaded xgb_pipeline
            fitted_xgb_model = xgb_pipeline_loaded.named_steps['regressor']
            # 4. Predict using the fitted XGBoost model directly
            xgb_pred_value = fitted_xgb_model.predict(input_df_transformed)[0]
            # --- End XGBoost Workaround ---

            model_predictions['xgb'] = xgb_pred_value
            predictions_list.append(model_predictions['xgb'])

        except Exception as predict_err:
             print(f"Error during model prediction step: {predict_err}")
             print(traceback.format_exc())
             # Check if the loaded model objects are valid
             print(f"ET Pipeline Loaded: {et_pipeline_loaded}")
             print(f"RF Pipeline Loaded: {rf_pipeline_loaded}")
             print(f"LR Pipeline Loaded: {lr_pipeline_loaded}")
             print(f"XGB Pipeline Loaded: {xgb_pipeline_loaded}")
             return jsonify({'status': 'error', 'message': f'Prediction engine error: {predict_err}'}), 500

        # --- Ensemble Calculation (No Change) ---
        raw_ensemble_pred = np.mean(predictions_list)

        # --- Map Raw Prediction to Closest Fibonacci (No Change) ---
        final_fibonacci_effort = find_closest_fibonacci(raw_ensemble_pred)

        # --- Logging (No Change) ---
        log_preds_str = f"ET: {model_predictions['et']:.2f}, RF: {model_predictions['rf']:.2f}, LR: {model_predictions['lr']:.2f}, XGB: {model_predictions['xgb']:.2f}"
        print(f"Individual Predictions: {log_preds_str}")
        print(f"Raw Ensemble Prediction: {raw_ensemble_pred:.2f}")
        print(f"Final Fibonacci Effort: {final_fibonacci_effort}")

        # --- JSON Response (No Change) ---
        return jsonify({
            'status': 'success',
            'raw_prediction': round(raw_ensemble_pred, 2),
            'fibonacci_effort': final_fibonacci_effort
        })

    # Catch specific validation errors or general exceptions (No Change)
    except ValueError as ve:
        print(f"Data Validation Error: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        print(f"!!!!!!!! Unexpected Error during prediction endpoint !!!!!!!!!!")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'An unexpected server error occurred.'}), 500

# --- Run the Flask App (No Changes) ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)