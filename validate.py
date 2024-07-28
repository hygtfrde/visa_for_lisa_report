import pandas as pd

file_path = 'Visa_For_Lisa_Loan_Modelling.csv'
df = pd.read_csv(file_path)

user_input = input("Enter the column name to check for min and max values: ")

if user_input in df.columns:
    min_value = df[user_input].min()
    max_value = df[user_input].max()
    
    print(f"The minimum value in the '{user_input}' column is: {min_value}")
    print(f"The maximum value in the '{user_input}' column is: {max_value}")
else:
    print(f"Column '{user_input}' does not exist in the DataFrame.")
