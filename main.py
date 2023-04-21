import streamlit as st
import pandas as pd
import subprocess

st.set_page_config(page_title="Simple Streamlit app", page_icon=":whale:")

# load css
st.markdown(
    "<style> .centered {text-align: center;} </style>", unsafe_allow_html=True
)

# Apply CSS class to title
st.markdown(
    '<h1 class="centered">Simple Streamlit App</h1>', unsafe_allow_html=True
)

# Upload a CSV file
st.write("## Upload a CSV file")
uploaded_file = st.file_uploader("Select a csv file", type="csv")

if uploaded_file is not None:
    # Create a list of possible delimiters
    delimiters = {",": ",", ";": ";", ".": ".", "Tab": "\tab"}
    delimiter = st.selectbox("Select delimiter:", list(delimiters.keys()))

    df = pd.read_csv(uploaded_file, sep=delimiter, engine='python')
    st.write(df)

    # Retrieve all columns in DataFrame and store them in list
    column_names = df.columns.tolist()

    if len(column_names) <= 1:
        st.error("DataFrame error! Please choose a different delimiter.")
    else:
        # Select output
        st.write("## Select output")
        output_path = st.selectbox("Select output", column_names)

        # Select input checkboxes
        st.write("## Select input")
        selected_columns = output_path

        for column_name in column_names:
            if column_name not in selected_columns:
                selected = st.checkbox(column_name)

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
                color: white;
                padding: 15px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 18px;
                margin: 4px 2px;
                cursor: pointer;
            }
            </style>
        """

        st.markdown(button_style, unsafe_allow_html=True)

        if st.button("Run"):
            output = run_external_script()
            st.write(output)
