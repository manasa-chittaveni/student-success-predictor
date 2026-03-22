
🎓 Student Success Predictor
Overview
A deep learning project that predicts student performance (High, Medium, Low) based on academics, communication skills, teamwork, coding ability, and creativity.
This tool is interactive: students enter their own profile and instantly receive a prediction. It also tracks results and visualizes performance distributions.

Dataset
File: student_success.csv

Columns:

Marks (0–100)

Attendance (0–100)

Communication (1–10)

Teamwork (1–10)

Coding Skill (1–10)

Creativity (1–10)

Overall Performance (High/Medium/Low)

Model
Framework: PyTorch

Architecture: Multi-Layer Perceptron with two hidden layers

Loss Function: CrossEntropyLoss

Optimizer: Adam

Training
The model is trained for 50 epochs. Loss decreases steadily, showing effective learning.
Example log:

Code
Epoch 10/50, Loss: 1.1089
Epoch 20/50, Loss: 0.9946
Epoch 30/50, Loss: 0.8712
Epoch 40/50, Loss: 0.7300
Epoch 50/50, Loss: 0.5864
Test Accuracy: 100.00%
Interactive Tool
After training, the script prompts for student details:

Code
Enter marks (0-100): 82
Enter attendance percentage (0-100): 88
Enter communication skill (1-10): 7
Enter teamwork skill (1-10): 8
Enter coding skill (1-10): 9
Enter creativity skill (1-10): 6
Predicted Performance: High
Multiple students can be entered in one run.

Tracker
All inputs and predictions are saved into predictions.csv.
Append mode ensures each run adds new rows, building a history of predictions.

Dashboard
Two visualizations are generated:

Training Loss Curve

Predicted Performance Distribution

Future enhancements include pie charts for skill averages, correlation heatmaps, and interactive dashboards.

How to Run
bash
git clone https://github.com/manasa-chittaveni/student-success-predictor.git
cd student-success-predictor
pip install -r requirements.txt
python student_success.py
Results
Accuracy: ~100% on synthetic dataset

Predictions saved in predictions.csv

Visualizations generated automatically

Future Work
Expand dataset with diverse student profiles

Add features like leadership, problem-solving, extracurriculars

Deploy as a web app using Flask or Streamlit

Integrate real-world datasets such as UCI Student Performance