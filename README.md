This is a better version of the previous project in this from a data of 10000 students we filtered the data and analysed the data of 4000+ students and made a prediction about their placement.
Student Placement Probability Predictor

An interactive web application powered by Streamlit and Machine Learning that calculates a student's likelihood of securing a job placement. This tool uses a 6-feature predictive model to deliver instant analysis alongside dynamic session data tracking.

Live App Link: https://newmlproject.streamlit.app/

Features

Instant Predictions: Input key student metrics and get a localized probability percentage instantly.
Smart Analysis Alerts: Dynamic feedback panels change color (Success/Warning/Error) depending on the calculated threshold.
In-Memory Session Logging: Tracks all inputs and predictions made during your current browser session.
Instant Data Export: Download your entire session's prediction history directly as a .csv file via the sidebar interface—no external databases required!

Tech Stack

Frontend UI: Streamlit
Data Processing: Pandas, NumPy
Machine Learning Inference: Scikit-Learn (via pickle)
Project2/
├── .gitignore               Tells Git which files to ignore (cache, local files)
├── app.py                  Main Streamlit application code
├── model.pkl               Trained 6-feature ML model weights
├── requirements.txt        Python package dependencies
└── README.md               Project documentation

Local Installation and Setup

Follow these quick steps to run the predictor pipeline locally on your computer:

    Clone the Repository
    git clone https://github.com/MehulSahai410/NewMLproject.git
    cd newMLproject

    Install Dependencies
    Ensure you have Python installed, then install the necessary application modules using pip:
    pip install -r requirements.txt

    Run the App
    Launch the local web server to run the interface on your localhost environment:
    streamlit run app.py
