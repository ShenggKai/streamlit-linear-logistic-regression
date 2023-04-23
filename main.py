import streamlit as st
import pandas as pd
from logistic import get_result_lg
from linear import get_result_ln

st.set_page_config(page_title="Simple Streamlit App", page_icon=":whale:")

# load css
st.markdown(
    "<style> .centered {text-align: center;color: #00B3B3;} </style>",
    unsafe_allow_html=True,
)

# Apply CSS class to title
st.markdown(
    '<h1 class="centered">Simple Streamlit App</h1>', unsafe_allow_html=True
)
st.write("")
st.write("")

# Upload a CSV file
st.write("## Upload a CSV file")
uploaded_file = st.file_uploader("Select a csv file", type="csv")

if uploaded_file is not None:
    # Create a list of possible delimiters
    delimiters = {",": ",", ";": ";", ".": ".", "Tab": "\tab"}
    delimiter = st.selectbox("Select delimiter:", list(delimiters.keys()))
    df = pd.read_csv(uploaded_file, sep=delimiter, engine="python")
    # st.write(df)

    # Retrieve all columns in DataFrame and store them in list
    column_names = df.columns.tolist()
    if len(column_names) <= 1:
        st.error("DataFrame error! Please choose a different delimiter.")
    else:
        # reprocessing dataset
        if not all(
            pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
        ):
            # get the column names of non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
            # one-hot encode non-numeric columns
            df = pd.get_dummies(df, columns=non_numeric_cols)
            column_names = df.columns.tolist()

        st.write(df)

        # Select output
        st.write("## Select output")

        # list to store output
        selected_output = []
        selected_output.append(st.selectbox("Select output", column_names))

        # Select input checkboxes
        st.write("## Select input")
        checkbox_values = []  # List to store all checkboxes value
        checkbox_status = []  # List to store all checkboxes status
        for column_name in column_names:
            if column_name not in selected_output:
                check_box = st.checkbox(column_name)
                checkbox_status.append(check_box)
                checkbox_values.append(column_name)

        # list to store checked boxes aka input
        selected_input = []
        for i in range(len(checkbox_values)):
            if checkbox_status[i]:
                selected_input.append(checkbox_values[i])

        # Custom random state
        st.write("## Split dataset")
        random_state = st.number_input(
            "Enter random state number between 0 and $2^{32} - 1$:",
            step=1,
            value=19521338,
            min_value=0,
            max_value=2**32 - 1,
        )

        # Split dataset
        # Define the initial values
        test_size = 20
        train_size = 80

        # Create a slider for the first value
        st.text("")
        st.write("Choose test size to evaluate model")
        test_size = st.slider("Test size (%)", 1, 99, test_size)

        # Calculate the second value based on the slider value
        train_size = 100 - test_size
        st.write(f"- Train size: {str(train_size)}%")
        st.write(f"- Test size: {str(test_size)}%")

        test_size = float(test_size / 100)

        # Select algorithm
        st.write("## Select algorithm")
        algorithm = st.selectbox(
            "Select algorithm",
            [
                "Logistic regression",
                "Linear regression",
            ],
        )

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
            # check if at least one checkbox is checked
            if len(selected_input) == 0:
                st.warning("Please select input!")
            else:
                st.write("## Result")

                # Return result based on algorithm
                result, data_shape = [], []
                result_train, result_test = [], []
                fig = None
                flag_error = False
                if algorithm == "Logistic regression":
                    try:
                        result, data_shape, fig = get_result_lg(
                            df,
                            selected_input,
                            selected_output,
                            test_size,
                            random_state,
                        )
                    except Exception as e:
                        st.error(f"Error occurred: {e}")
                        # set flag to True if an error occurred
                        flag_error = True

                # algorithm is linear regression
                else:
                    try:
                        result_train, result_test, data_shape = get_result_ln(
                            df,
                            selected_input,
                            selected_output,
                            test_size,
                            random_state,
                        )
                    except Exception as e:
                        st.error(f"Error occurred: {e}")
                        # set flag to True if an error occurred
                        flag_error = True

                if not flag_error and algorithm == "Logistic regression":
                    train_shape = str(data_shape[0]) + " " + str(data_shape[1])
                    test_shape = str(data_shape[2]) + " " + str(data_shape[3])
                    # code to run if no errors occurred
                    table_result = {
                        "": ["Training set", "Test set"],
                        "Data shape": [train_shape, test_shape],
                        "F1 score": [result[0], result[1]],
                    }
                    st.table(table_result)

                    # Confusion matrix
                    st.text("")
                    st.pyplot(fig)

                elif not flag_error and algorithm == "Linear regression":
                    train_shape = str(data_shape[0]) + " " + str(data_shape[1])
                    test_shape = str(data_shape[2]) + " " + str(data_shape[3])
                    table_result = {
                        "": ["Training set", "Test set"],
                        "Data shape": [train_shape, test_shape],
                        "Accuracy": [result_train[0], result_test[0]],
                        "MAE": [result_train[1], result_test[1]],
                        "MSE": [result_train[2], result_test[2]],
                        "RMSE": [result_train[3], result_test[3]],
                    }
                    st.table(table_result)
