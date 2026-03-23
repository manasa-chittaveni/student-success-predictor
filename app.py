import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load dataset for preprocessing
df = pd.read_csv("student.csv")
encoder = LabelEncoder()
df['overall_performance'] = encoder.fit_transform(df['overall_performance'])

features = ['marks','attendance','communication','teamwork','coding_skill','creativity']
X = df[features].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 2: Define model (same as training)
class StudentSuccessModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StudentSuccessModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Load trained model
model = StudentSuccessModel(input_dim=6, hidden_dim=32, output_dim=3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Step 4: Streamlit UI
st.title("🎓 Student Success Predictor")

marks = st.slider("Marks (0-100)", 0, 100, 75)
attendance = st.slider("Attendance (%)", 0, 100, 80)
communication = st.slider("Communication (1-10)", 1, 10, 7)
teamwork = st.slider("Teamwork (1-10)", 1, 10, 8)
coding_skill = st.slider("Coding Skill (1-10)", 1, 10, 9)
creativity = st.slider("Creativity (1-10)", 1, 10, 6)

if st.button("Predict Performance"):
    input_data = [[marks, attendance, communication, teamwork, coding_skill, creativity]]
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    st.success(f"Predicted Performance: {encoder.inverse_transform([prediction])[0]}")
