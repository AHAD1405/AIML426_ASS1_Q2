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

def FilterGA(feature, target):
    """
        CORRELATION CLACULATION: The corrwith method is used to calculate the correlation between each feature in X and the target variable y. 
            This produces a series where each feature's correlation with the target is listed.

        THRESHOLD: A threshold defined is 0.5, and only features with an absolute correlation greater than this threshold are selected.
            This means we're interested in features that are strongly correlated with the target variable.
        
        DATA TRANSFORMATION: The selected features' names are extracted based on the threshold condition and stored in selected_features.
            A new DataFrame X_selected is created, containing only the selected features that met the correlation threshold
    """
    threshold = 0.2

    # Calculate the correlation between each feature and the target
    correlations = feature.corrwith(target)

    # Select features that have an absolute correlation value above the threshold
    selected_features = correlations[abs(correlations) > threshold].index.tolist()

    # Create a new DataFrame with only the selected features
    #dataset_selected = dataset[selected_features]
    return selected_features

def initial_pop(population_size, feature_num, seed_val):
    """
    Generate the initial population for the genetic algorithm.

    Args:
        population_size (int): The desired size of the initial population.
        feature_num (int): The number of the feature in dataset.

    Returns:
        list: A list of individuals, where each individual is a binary vector
              representing a potential solution. The length of each individual is equal to `feature_num`.
    """
    random.seed(seed_val)
    return [np.random.randint(2, size=feature_num) for _ in range(population_size)]

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

    for run in range(run_no):
        start_time = time.time()  # Start timer for this run

        # FEATURE FITNESS FUNCTION: 
        FilterGA_fs = FilterGA(feature, target)   # Filter-based feature selection

        # DATA TASFORMATION: create new dataset for each filter approach, created datset that only contain selected feature 
        FilterGA_dataset = dataset[:][FilterGA_fs + ['class']]

        # INITIALIZATION: Initialize the population 
        FilterGA_population = initial_pop(population_size, FilterGA_dataset.iloc[:,:-1].shape[1], seed_value[run])



if __name__ == "__main__":
    main()