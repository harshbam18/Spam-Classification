import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset and test the models
df = pd.read_csv(r'C:\Users\Harsh Sharma\Desktop\miniproject\SPAM text message 20170820 - Data.csv')
fig = px.histogram(df, x="Category", color="Category", color_discrete_sequence=["#871fff","#ffa78c"])
fig.show()
a=df['Message']
b=df['Category']
a_train, a_test, b_train, b_test = train_test_split( a, b,  test_size=0.20 , random_state=27)
pipeMNB = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB())])
pipeCNB = Pipeline([('tfidf', TfidfVectorizer()),('clf', ComplementNB())])
pipeSVCt = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
#MultinomialNB
pipeMNB.fit(a_train, b_train)
predictMNB = pipeMNB.predict(a_test)
#ComplementNB
pipeCNB.fit(a_train, b_train)
predictCNB = pipeCNB.predict(a_test)
#LinearSVC
pipeSVCt.fit(a_train, b_train)
predictSVCt = pipeSVCt.predict(a_test)
print(f"CNB: {accuracy_score(b_test, predictMNB):.4f}")
print(f"MNB: {accuracy_score(b_test, predictCNB):.4f}")
print(f"SVC: {accuracy_score(b_test, predictSVCt):.4f}")
import seaborn as sns
classifiers = ['MultinomialNB', 'ComplementNB', 'LinearSVC']
accuracies = [accuracy_score(b_test, pipeMNB.predict(a_test)),
              accuracy_score(b_test, pipeCNB.predict(a_test)),
              accuracy_score(b_test, predictSVCt)]

plt.figure(figsize=(10, 6))
sns.barplot(x=classifiers, y=accuracies)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Classifier Comparison')
plt.show()
from sklearn.metrics import roc_curve, auc

# Map 'ham' to 0 and 'spam' to 1
b_test_binary = b_test.map({'ham': 0, 'spam': 1})

# Calculate ROC curve for LinearSVC
fpr, tpr, _ = roc_curve(b_test_binary, pipeSVCt.decision_function(a_test))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - LinearSVC')
plt.legend(loc='lower right')
plt.show()
# Sidebar
st.sidebar.title("Spam Detection App")

# Show the first few rows of the dataset in the main area
st.title("Spam Detection Web App")
if st.checkbox("Show raw data"):
    st.write(df.head())
    st.write(df.tail())

# Data preprocessing
M = df['Message']
C = df['Category']
M_train, M_test, C_train, C_test = train_test_split(M, C, test_size=0.20, random_state=27)

# Model training
pipeSVC = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
pipeSVC.fit(M_train, C_train)
# Web app for user input
st.header("Enter a message to check for spam:")
user_input = st.text_area("Type your message here:")

# Prediction
if st.button("Check for Spam"):
    if user_input:
        result = pipeSVC.predict([user_input])
        st.success(f"Result: {result[0]}")
    else:
        st.warning("Please enter a message.")
