import pandas as pd

def detect_outliers(df, column_name):
    if column_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[column_name]):
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
            num_outliers = len(outliers)
            num_total = len(df)
            
            percent_outliers = (num_outliers / num_total) * 100
            percent_regular = 100 - percent_outliers

            return percent_regular, percent_outliers, lower_bound, upper_bound
        else:
            print(f"The column '{column_name}' is not numerical and cannot have outliers detected.")
            return None, None, None, None
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return None, None, None, None

def main():
    file_path = 'Visa_For_Lisa_Loan_Modelling.csv'
    df = pd.read_csv(file_path)

    while True:
        column_name = input("Enter the column name to check for outliers: ").strip()
        if column_name in df.columns:
            percent_regular, percent_outliers, lower_bound, upper_bound = detect_outliers(df, column_name)
            if percent_regular is not None and percent_outliers is not None:
                print(f"Percentage of data points in the regular range: {percent_regular:.2f}%")
                print(f"Percentage of data points in outliers: {percent_outliers:.2f}%")
                print(f"Lower bound of regular range: {lower_bound:.2f}")
                print(f"Upper bound of regular range: {upper_bound:.2f}")
            break
        else:
            print(f"Column '{column_name}' does not exist in the DataFrame. Please enter a valid column name.")

if __name__ == "__main__":
    main()
