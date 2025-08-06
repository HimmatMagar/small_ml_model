from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pipeline
pipeline = joblib.load('model.pkl')

# Define input columns in order
all_columns = ['time_spent_alone', 'social_event_attendance', 'going_outside',
               'friends_circle_size', 'post_frequency',
               'drained_after_socializing', 'stage_fear']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_data = {}
        for col in all_columns:
            input_data[col] = request.form.get(col, type=float)

        df = pd.DataFrame([input_data])

        # Predict using the pipeline
        result = pipeline.predict(df)[0]
        label_map = {
            0: "You are Extrovert",
            1: "You are Introvert"
        }
        prediction = label_map.get(result, f"Unknown Prediction: {result}")
        # prediction = f"Prediction: {result}"

    return render_template('index.html', columns=all_columns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
