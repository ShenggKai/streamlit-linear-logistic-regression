import streamlit as st
import pandas as pd
import subprocess

st.title("Upload CSV File")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.write(df)

# Create text box labeled "Select input"
st.text("Select input")
if "checked_count" not in st.session_state:
    st.session_state.checked_count = 0

age_checkbox = st.checkbox("Age")
salary_checkbox = st.checkbox("EstimatedSalary")
purchased_checkbox = st.checkbox("Purchased")

# Check if maximum number of checkboxes have been checked
if age_checkbox:
    st.session_state.checked_count += 1
if salary_checkbox:
    st.session_state.checked_count += 1
if purchased_checkbox:
    st.session_state.checked_count += 1

if st.session_state.checked_count > 2:
    st.warning("You can only check up to 2 options.")
    if age_checkbox:
        age_checkbox = False
        st.session_state.checked_count -= 1
    if salary_checkbox:
        salary_checkbox = False
        st.session_state.checked_count -= 1
    if purchased_checkbox:
        purchased_checkbox = False
        st.session_state.checked_count -= 1

# Create text box labeled "Select output"
output_path = st.selectbox(
    "Select output", ["Age", "EstimatedSalary", "Purchased"]
)

# Create dropbox labeled "Select algorithm"
algorithm = st.selectbox(
    "Select algorithm",
    [
        "Logistic Regression with normalized data",
        "Logistic Regression without normalized data",
        "Algorithm 3",
    ],
)


# Define function to run external Python file
def run_external_script():
    result = subprocess.run(
        ["python", "logistic.py"],
        capture_output=True,
        text=True,
    )
    return result.stdout


# Create a Streamlit button that runs the external script
if st.button("Run"):
    output = run_external_script()
    st.write(output)
