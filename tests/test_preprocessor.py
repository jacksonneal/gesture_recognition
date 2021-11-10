from preprocessing.Preprocessor import Preprocessor


def test_preprocessor_join():
    Preprocessor.join_datasets("datasets/initial", "datasets/joined/joined.csv")
    assert True
