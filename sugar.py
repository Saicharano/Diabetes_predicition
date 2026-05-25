import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
df = pd.read_csv("diabetes.csv")
columns = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in columns:
    median = df[df[col]!=0][col].median()
    df[col]=df[col].replace(0,median)
x = df.drop("Outcome",axis=1)
y = df["Outcome"]
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.2,
                                                 random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
model = LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
class_report = classification_report(y_pred,y_test)
conf_mat = confusion_matrix(y_pred,y_test)
print(class_report)
print(conf_mat)
print(accuracy)