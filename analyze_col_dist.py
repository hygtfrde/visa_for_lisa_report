import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_column_distribution(df, column_name, column_type):
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in DataFrame.")
        return

    column_data = df[column_name]

    if column_type == 'numerical':
        analyze_numerical_column(column_data)
    elif column_type == 'categorical':
        analyze_categorical_column(column_data)
    else:
        print(f"Invalid column type '{column_type}'. Please specify 'numerical' or 'categorical'.")

def analyze_numerical_column(column_data):
    mean = column_data.mean()
    std_dev = column_data.std()
    skewness = column_data.skew()
    kurtosis = column_data.kurtosis()

    plt.hist(column_data, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"Histogram of {column_data.name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    description = f"Column '{column_data.name}' Distribution:\n"
    description += f"Mean: {mean}\n"
    description += f"Standard Deviation: {std_dev}\n"
    description += f"Skewness: {skewness} (indicates asymmetry)\n"
    description += f"Kurtosis: {kurtosis} (indicates tailedness)\n"

    if abs(skewness) > 1:
        description += "The distribution is highly skewed.\n"
    elif abs(skewness) > 0.5:
        description += "The distribution is moderately skewed.\n"
    else:
        description += "The distribution is approximately symmetric.\n"

    print(description)

def analyze_categorical_column(column_data):
    value_counts = column_data.value_counts()
    mode = column_data.mode().values[0]
    mode_freq = value_counts.max()
    total_count = len(column_data)

    value_counts.plot(kind='bar', edgecolor='k', alpha=0.7)
    plt.title(f"Bar Chart of {column_data.name}")
    plt.xlabel("Category")
    plt.ylabel("Frequency")
    plt.show()

    description = f"Column '{column_data.name}' Distribution:\n"
    description += f"Most common value: {mode} (Frequency: {mode_freq})\n"

    if mode_freq / total_count > 0.5:
        description += "The majority of values cluster around the most common category.\n"
    elif mode_freq / total_count > 0.2:
        description += "A significant portion of values cluster around the most common category.\n"
    else:
        description += "Values are relatively well-distributed across categories.\n"

    print(description)

def main():
    df = pd.read_csv('Visa_For_Lisa_Loan_Modelling.csv')

    while True:
        column_type = input("Enter the type of column to analyze ('1: numerical' or '2: categorical'): ").strip()
        if column_type == '1':
            column_type = 'numerical'
            break
        elif column_type == '2':
            column_type = 'categorical'
            break
        else:
            print("Invalid input. Please enter '1' for numerical or '2' for categorical.")

    while True:
        column_name = input("Enter the column name to analyze: ").strip()
        if column_name in df.columns:
            break
        else:
            print(f"Invalid column name. Please enter one of the following: {', '.join(df.columns)}")

    analyze_column_distribution(df, column_name, column_type)
if __name__ == "__main__":
    main()
