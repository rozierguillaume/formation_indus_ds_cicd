import pandas as pd
import numpy as np

from src.feature_engineering import names, age_impute


def test_names_should_return_dataframe_without_Name_column():
    # Given
    train = pd.DataFrame(
        {
            "Name":
                ['Braund, Mr. Owen Harris',
                 'Cumings, Mrs. John Bradley '
                 '(Florence Briggs Thayer)',
                 'Heikkinen, Miss. Laina',
                 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                 'Allen, Mr. William Henry']
        })
    test = pd.DataFrame(
        {
            "Name":
                ['Kelly, Mr. James',
                 'Wilkes, Mrs. James (Ellen Needs)',
                 'Myles, Mr. Thomas Francis',
                 'Wirz, Mr. Albert',
                 'Hirvonen, Mrs. Alexander (Helga E Lindqvist)']
        })

    # When
    new_train, new_test = names(train, test)

    # Then
    assert isinstance(new_train, pd.DataFrame)
    assert isinstance(new_test, pd.DataFrame)
    assert "Name" not in new_train.columns
    assert "Name" in new_test.columns


def test_names_should_return_dataframe_with_name_lenght():
    # Given
    train = pd.DataFrame(
        {
            "Name":
                ['Braund, Mr. Owen Harris',
                 'Cumings, Mrs. John Bradley '
                 '(Florence Briggs Thayer)',
                 'Heikkinen, Miss. Laina',
                 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                 'Allen, Mr. William Henry']
        })
    expected_train_length = [23, 51, 22, 44, 24]
    test = pd.DataFrame(
        {
            "Name":
                ['Kelly, Mr. James',
                 'Wilkes, Mrs. James (Ellen Needs)',
                 'Myles, Mr. Thomas Francis',
                 'Wirz, Mr. Albert',
                 'Hirvonen, Mrs. Alexander (Helga E Lindqvist)']
        })
    expected_test_lenght = [16, 32, 25, 16, 44]

    # When
    new_train, new_test = names(train, test)

    # Then
    assert "Name_Len" in new_train.columns
    assert "Name_Len" in new_test.columns
    assert new_train["Name_Len"].tolist() == expected_train_length
    assert new_test["Name_Len"].tolist() == expected_test_lenght


def test_names_should_return_dataframe_with_title():
    # Given
    train = pd.DataFrame(
        {
            "Name":
                ['Braund, Mr. Owen Harris',
                 'Cumings, Mrs. John Bradley '
                 '(Florence Briggs Thayer)',
                 'Heikkinen, Miss. Laina',
                 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                 'Allen, Mr. William Henry']
        })
    expected_train_titles = ["Mr.", "Mrs.", "Miss.", "Mrs.", "Mr."]
    test = pd.DataFrame(
        {
            "Name":
                ['Kelly, Mr. James',
                 'Wilkes, Mrs. James (Ellen Needs)',
                 'Myles, Mr. Thomas Francis',
                 'Wirz, Mr. Albert',
                 'Hirvonen, Mrs. Alexander (Helga E Lindqvist)']
        })
    expected_test_titles = ["Mr.", "Mrs.", "Mr.", "Mr.", "Mrs."]

    # When
    new_train, new_test = names(train, test)

    # Then
    assert "Name_Title" in new_train.columns
    assert "Name_Title" in new_test.columns
    assert new_train["Name_Title"].tolist() == expected_train_titles
    assert new_test["Name_Title"].tolist() == expected_test_titles


def test_age_impute_should_return_dataframe_with_age_and_age_null_flag_column():
    # Given
    train = pd.DataFrame(
        {
            "Age": [12, 52, 23, np.nan, 42],
            "Name_Title": ["Mr.", "Mrs.", "Mr.", "Mr.", "Mrs."],
            "Pclass": [0, 1, 2, 1, 0]
        })
    test = pd.DataFrame(
        {
            "Age": [7, 14, 120, 31, np.nan],
            "Name_Title": ["Mr.", "Mrs.", "Miss.", "Mrs.", "Mr."],
            "Pclass": [2, 0, 2, 1, 0]
        })

    # When
    new_train, new_test = age_impute(train, test)

    # Then
    for output in (new_train, new_test):
        assert isinstance(output, pd.DataFrame)
        assert "Age" in output.columns
        assert "Age_Null_Flag" in output.columns


def test_age_impute_should_return_dataframe_with_no_null_in_age_column():
    # Given
    # note : at least one pair Title-Pclass covering the null case should exist
    # in the train set
    train = pd.DataFrame(
        {
            "Age": [12, 52, 23, np.nan, 42],
            "Name_Title": ["Mr.", "Mrs.", "Mr.", "Mr.", "Mrs."],
            "Pclass": [0, 1, 2, 2, 0]
        })
    test = pd.DataFrame(
        {
            "Age": [7, 14, 120, 31, np.nan],
            "Name_Title": ["Mr.", "Mrs.", "Miss.", "Mrs.", "Mr."],
            "Pclass": [2, 0, 2, 1, 0]
        })

    # When
    new_train, new_test = age_impute(train, test)

    # Then
    for output in (new_train, new_test):
        print(output["Age"])
        assert output["Age"].notnull().all()


def test_age_impute_should_return_dataframe_binary_age_null_flag():
    # Given
    train = pd.DataFrame(
        {
            "Age": [12, 52, 23, np.nan, 42],
            "Name_Title": ["Mr.", "Mrs.", "Mr.", "Mr.", "Mrs."],
            "Pclass": [0, 1, 2, 1, 0]
        })
    test = pd.DataFrame(
        {
            "Age": [7, 14, 120, 31, np.nan],
            "Name_Title": ["Mr.", "Mrs.", "Miss.", "Mrs.", "Mr."],
            "Pclass": [2, 0, 2, 1, 0]
        })

    # When
    new_train, new_test = age_impute(train, test)

    # Then
    for output in (new_train, new_test):
        assert set(output["Age_Null_Flag"].unique()) == {0, 1}


def test_age_impute_should_flag_null_values_in_age_column():
    # Given
    train = pd.DataFrame(
        {
            "Age": [12, 52, 23, np.nan, 42],
            "Name_Title": ["Mr.", "Mrs.", "Mr.", "Mr.", "Mrs."],
            "Pclass": [0, 1, 2, 1, 0]
        })
    test = pd.DataFrame(
        {
            "Age": [7, 14, 120, 31, np.nan],
            "Name_Title": ["Mr.", "Mrs.", "Miss.", "Mrs.", "Mr."],
            "Pclass": [2, 0, 2, 1, 0]
        })

    # When
    new_train, new_test = age_impute(train, test)

    # Then
    assert new_train["Age_Null_Flag"].tolist() == [0, 0, 0, 1, 0]
    assert new_test["Age_Null_Flag"].tolist() == [0, 0, 0, 0, 1]
