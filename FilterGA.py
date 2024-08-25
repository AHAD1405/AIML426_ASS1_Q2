import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

def load_data(column_file_name, data_file_name):
    """
        Extract data from file and return datset
    """
    # Extract Column names --------------------------------------
    columns_name = []
    with open(column_file_name,'r') as columns_file: 
        columns = columns_file.readlines()
        for idx, line in enumerate(columns): # extract values
            if idx == 0: continue
            x = line.split()
            columns_name.append(x[0])
    columns_name.append('class')   # add column for target column

    # Extract data from file and create dataset -----------------
    dataset = pd.read_csv(data_file_name)
    dataset.columns = columns_name

    return dataset

def main():
    # Genetic operation setting 
    population_size = 50
    generations = 2
    mutation_rate = 0.2
    crossover_rate = 0.5
    seed_value = [20, 30, 40, 50, 60]
    run_no = 5
    competition_time_li = []
    acc_li = []
    # dataset file name
    data_file =  'sonar.data'
    column_file = 'sonar.names'

    # load data
    dataset = load_data(column_file, data_file)
    feature = dataset.iloc[:,:-1]
    target = dataset.iloc[:,-1]



if __name__ == "__main__":
    main()