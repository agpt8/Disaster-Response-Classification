import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data function:
    1. read input csv file containing messages
    2. read input csv file containing categories
    3. merge the messages and categories datasets using the common id
    4. assign this combined dataset to df
    
    Args:
        messages_filepath (str): path to messages   csv file
        categories_filepath (str): path to categories csv file
    Returns:
        df (pd.Dataframe): Return merged data as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df


def clean_data(df):
    """
    Clean Data function
    1. split categories into separate category columns
    2. convert category values to just numbers 0 or 1

    Args:
        df (pd.Datafame): raw Pandas DataFrame
    Returns:
        df (pd.Datafame): cleaned Pandas DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    # extract a list of new column names for categories.
    # rename the columns of `categories`
    firstrow = categories.loc[0, :].values
    category_colnames = [x[:-2] for x in firstrow]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database
    
    Args:
        df (pd.Dataframe): Cleaned data Pandas DataFrame
        database_filename (str): SQLite database file (.db) destination path
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_categories', engine, index=False)
    pass


def main():
    """
    This function implement the ETL pipeline:
        1. data extraction from .csv
        2. data cleaning and pre-processing
        3. data loading to SQLite database
    """
    print(sys.argv)

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n MESSAGES: {messages_filepath}\n CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepath of the messages and categories datasets as the first and second argument '
              'respectively, as well as the filepath of the database to save the cleaned data to as the third '
              'argument. \n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
