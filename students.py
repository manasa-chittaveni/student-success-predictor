import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("student.csv")

# Step 2: Preprocessing
encoder = LabelEncoder()
df['overall_performance'] = encoder.fit_transform(df['overall_performance'])

features = ['marks','attendance','communication','teamwork','coding_skill','creativity']
X = df[features].values
y = df['overall_performance'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 3: Define model
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

model = StudentSuccessModel(input_dim=6, hidden_dim=32, output_dim=3)

# Step 4: Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Step 5: Evaluation
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

# Step 6: Interactive Multiple Student Input + Tracker
results = []

while True:
    print("\n--- Student Performance Prediction Tool ---")
    marks = float(input("Enter marks (0-100): "))
    attendance = float(input("Enter attendance percentage (0-100): "))
    communication = float(input("Enter communication skill (1-10): "))
    teamwork = float(input("Enter teamwork skill (1-10): "))
    coding_skill = float(input("Enter coding skill (1-10): "))
    creativity = float(input("Enter creativity skill (1-10): "))

    user_data = [[marks, attendance, communication, teamwork, coding_skill, creativity]]
    user_data = scaler.transform(user_data)
    user_tensor = torch.tensor(user_data, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(user_tensor)
        predicted_class = torch.argmax(prediction).item()
        performance = encoder.inverse_transform([predicted_class])[0]
        print("Predicted Performance:", performance)

    # Save to tracker
    results.append({
        "marks": marks,
        "attendance": attendance,
        "communication": communication,
        "teamwork": teamwork,
        "coding_skill": coding_skill,
        "creativity": creativity,
        "predicted_performance": performance
    })

    again = input("Do you want to enter another student? (y/n): ").lower()
    if again != 'y':
        break

# Save results to CSV (append mode)
results_df = pd.DataFrame(results)
try:
    existing = pd.read_csv("predictions.csv")
    results_df = pd.concat([existing, results_df], ignore_index=True)
except FileNotFoundError:
    pass
results_df.to_csv("predictions.csv", index=False)
print("\nAll predictions appended to predictions.csv")

# Step 7: Dashboard Visualization
def show_dashboard(losses, results_df):
    plt.figure(figsize=(12,5))

    # Loss curve
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), losses, marker='o', color='blue')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Performance distribution
    plt.subplot(1,2,2)
    performance_counts = results_df['predicted_performance'].value_counts()
    performance_counts.plot(kind='bar', color=['green','orange','red'])
    plt.title("Predicted Performance Distribution")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.show()

show_dashboard(losses, results_df)
