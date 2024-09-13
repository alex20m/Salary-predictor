import pandas as pd

def create_data_sets(input_data):

    training_set_first_entry = 0
    training_set_last_entry = int(len(input_data) * 0.70)
    testing_set_first_entry = training_set_last_entry + 1
    testing_set_last_entry = int(training_set_last_entry + (len(input_data) - testing_set_first_entry) / 2)
    validating_set_first_entry = testing_set_last_entry + 1
    validating_set_last_entry = len(input_data) - 1

    print(training_set_first_entry)
    print(training_set_last_entry)
    print(testing_set_first_entry)
    print(testing_set_last_entry)
    print(validating_set_first_entry)
    print(validating_set_last_entry)

    training_data = input_data.loc[training_set_first_entry:training_set_last_entry]
    testing_data = input_data.loc[testing_set_first_entry:testing_set_last_entry]
    validating_data = input_data.loc[validating_set_first_entry:validating_set_last_entry]

    training_data.to_csv("training_data.csv", index=False)
    testing_data.to_csv("testing_data.csv", index=False)
    validating_data.to_csv("validating_data.csv", index=False)



def main():
    data = pd.read_csv("salary_data.csv")

    for entry in range(len(data['income'])):
        data.loc[entry, 'income'] = int(round(data['income'][entry]))
    data['income'] = data['income'].astype(int)

    for entry in range(len(data['gender'])):
        if data['gender'][entry] == 'M':
            data.loc[entry, 'gender'] = 0
        if data['gender'][entry] == 'F':
            data.loc[entry, 'gender'] = 1

    new_data = data.drop('ID', axis='columns').sample(frac=1).dropna(axis=0)
    new_data.to_csv("processed_salary_data.csv", index=False)

if __name__ == "__main__":
    main()
    create_data_sets(pd.read_csv("processed_salary_data.csv"))
