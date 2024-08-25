import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import time
import math

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

def normalize_data(dataset):
    # Initialize and fit the scaler to dataset
    scaler = MinMaxScaler() 
    normalized_data = scaler.fit_transform(dataset) 
    
    # Convert the normalized data back into a DataFrame
    normalized_dataset = pd.DataFrame(normalized_data, columns=dataset.columns) 

    return normalized_dataset

def classification(individual, dataset, target):
    """
        The fitness function evaluates the performance of the selected features:
    """
    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    if len(selected_features) == 0:
        return 0,
    
    X_selected = dataset[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    return accuracy_score(y_test, predictions)

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

def crossover(parent1, parent2, crossover_rate):
    """
        The function chooses a random crossover point, 
        then creates two new individuals by concatenating the genetic material of the two parents at that point
    """
    if np.random.rand() < crossover_rate:
        num_features = parent1.shape[0]
        crossover_point = np.random.randint(1, num_features - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    else:
        return parent1, parent2

def mutation(individual, mutation_rate):
    """
    The function iterates over each element of the individual's genetic vector and, 
    with probability mutation_rate, flips the element (i.e., changes 0 to 1 or 1 to 0)

    PARAM: 
        - individual (list): A list representing the genetic vector or chromosome of an individual in the population.
        - items: A list of tuples, where each tuple represents an item with its weight and value (weight, value).
    RETURN:
        - mutated_individual (list): A new list representing the mutated individual's genetic vector after applying the mutation operation.
    """
    gens = individual.shape[0]
    mutated_individual = individual.copy()
    for i in range(gens):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual

def create_table(FilterGA, mean_, std_):
    """
    Create a dataset with two columns from two input lists.

    Args:
        list1 (list): The first list of values.
        list2 (list): The second list of values.
        column1_name (str): The name of the first column.
        column2_name (str): The name of the second column.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the two columns.
    """
    # Check if the lists have the same length
    #if len(WrapperGA) != len(FilterGA):
    #    raise ValueError("The input lists must have the same length.")

    # First column
    first_column = ['Run 1','Run 2','Run 3','Run 4','Run 5']

    # Create a dictionary with the two lists as values
    data = {'': first_column, 'FilterGA': FilterGA}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    mean_row = pd.DataFrame({'': ['Mean'], 'FilterGA': [mean_]})
    data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'FilterGA': [std_] })
    data_table = pd.concat([data_table, std_row], ignore_index=True)

    return data_table

def calculate_stats(list_value):
    """
    Calculate the mean and standard deviation of a lists.

    Args:
        list_value (list): The first list of numbers.

    Returns:
        tuple: A tuple containing two elements:
            - mean1 (float): The mean of list.
            - std_dev1 (float): The standard deviation of list.
    """
    # Calculate the mean of list
    mean1 = sum(list_value) / len(list_value)

    # Calculate the standard deviation of list
    squared_diffs = [(x - mean1) ** 2 for x in list_value]
    variance1 = sum(squared_diffs) / len(list_value)
    std_dev1 = math.sqrt(variance1)

    return mean1, std_dev1

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

        # Apply Selection process
        for generation in range(generations):

            # EVALUATE: clasification and evaluates the performance of the selected features
            FilterGA_fitness = [classification(individual, FilterGA_dataset.iloc[:,:-1].values, FilterGA_dataset.iloc[:,-1].values) for individual in FilterGA_population]

            FilterGA_parents = np.array(FilterGA_population)[np.argsort(FilterGA_fitness)][-2:]

            # CROSSOVER: Perform crossover to generate new offspring
            FilterGA_offspring = crossover(FilterGA_parents[0], FilterGA_parents[1], crossover_rate)

            # MUTATION: Perform mutation on the new offspring
            FilterGA_offspring = [mutation(_, mutation_rate) for _ in FilterGA_offspring]

            # REPLACEMENT: Replace the least fit individuals with the new offspring
            FilterGA_population = np.vstack((FilterGA_population, FilterGA_offspring))

            FilterGA_population = FilterGA_population[np.argsort(FilterGA_fitness)][:-2]

            end_time = time.time()  # End timer for this run
        competitive_time = end_time - start_time  # Calculate competitive time for this run
        competition_time_li.append(competitive_time)
        acc_li.append(max(FilterGA_fitness))

    # Create a table and calculate mean and STD
    acc_mean, acc_std = calculate_stats(acc_li)
    acc_li = [round(x, 2) for x in acc_li]
    acc_table = create_table(acc_li, round(acc_mean), round(acc_std))

    competitive_time_mean, competitive_time_std = calculate_stats(competition_time_li)
    competition_time_li = [round(x, 2) for x in competition_time_li]
    competitive_time_table = create_table(competition_time_li, round(competitive_time_mean), round(competitive_time_std))
    
    # Print the best fitness values for each generation
    print(f'Accuracy Table\n {acc_table}')
    print(f'Competional Time Table:\n {competitive_time_table}')
    # print('Generation {}: FilterGA best fitness = {:.4f}, WrapperGA best fitness = {:.4f}'.format(generation, max(FilterGA_fitness), max(WrapperGA_fitness)))
    # print(competition_time_li)
    # dfsdfsdfs

if __name__ == "__main__":
    main()