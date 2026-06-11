import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st
@st.cache_resource
def train_model():
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
    return accuracy,class_report,conf_mat,model,scalar
accuracy,class_report,conf_mat,model,scalar = train_model()
st.title("Diabetes Report")
pregnancy = st.number_input("Enter the no. of pregnancy")
glucose = st.number_input("Enter the glucose level")
BloodPressure = st.number_input("Enter the Blood Pressure")
skinthickness = st.number_input("Enter the SkinThickness")
insulin = st.number_input("Enter the Insulin Level")
bmi = st.number_input("Enter the BMI: ")
Dpf = st.number_input("Enter the Diabetes pedigree function no.")
age = st.number_input("Enter the age")
input_data = [[
    pregnancy,
    glucose,
    BloodPressure,
    skinthickness,
    insulin,
    bmi,
    Dpf,
    age
]]
# lol
input_data = scalar.transform(input_data)
if st.button("predict"):
    prediction = model.predict(input_data)
    if prediction[0]==1:
        st.error("person is diabetic")
    else:
        st.success("person is Non - Diabatic")
