import streamlit as st
import pandas as pd
import subprocess

st.set_page_config(page_title="Streamlit simple app", page_icon=":whale:")

# load css
st.markdown(
    "<style> .centered {text-align: center;} </style>", unsafe_allow_html=True
)

# Apply CSS class to title
st.markdown(
    '<h1 class="centered">Streamlit Simple App</h1>', unsafe_allow_html=True
)

# Upload a CSV file
st.write("## Upload a CSV file")
uploaded_file = st.file_uploader("Select a csv file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.write(df)

    # Select output
    st.write("## Select output")
    output_path = st.selectbox(
        "Select output", ["Age", "EstimatedSalary", "Purchased"]
    )

    # Select input checkboxes
    st.write("## Select input")

    if output_path == "Age":
        salary_checkbox = st.checkbox("Salary")
        purchased_checkbox = st.checkbox("Purchased")
    elif output_path == "EstimatedSalary":
        age_checkbox = st.checkbox("Age")
        purchased_checkbox = st.checkbox("Purchased")
    else:  # Purchased
        age_checkbox = st.checkbox("Age")
        salary_checkbox = st.checkbox("Salary")

    # Select algorithm
    st.write("## Select algorithm")
    algorithm = st.selectbox(
        "Select algorithm",
        [
            "Linear regression",
            "Logistic regression",
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

    # Create button
    button_style = """
        <style>
        .stButton button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }
        </style>
    """

    st.markdown(button_style, unsafe_allow_html=True)

    if st.button("Run", help="Button help"):
        output = run_external_script()
        st.write(output)
