import cornac
import pandas as pd
import numpy as np

from cornac.eval_methods import BaseMethod
from recommenders.datasets.python_splitters import python_random_split
from recommenders.utils.constants import SEED

def run_bpr_model(data, k, epochs, learning_rate, train_size=0.8):
    """
    A function to train and evaluate bayesian personalised ranking recommender on the provided dataset.

    Args:
        data(pd.DataFrame): a dataframe with following columns - userID, itemID, rating
                            each userID and itemID has rating 1 if there is interaction else 0
        k(int): the value for top k products to rank and recommend
        epochs(int): max number of iterations for model training
        learning_rate(float): the learning rate for the bpr model
        train_size(float): amount of datasize to use for training
    
    Returns:
        test_result: All output metrics on test dataset for the trained bpr model
    """
    
    ## Split dataset into training and testing 
    train_data, test_data = python_random_split(data, train_size)

    ## Define evaluation object for train and test data
    eval_method = BaseMethod.from_splits(
                                            train_data=train_data.values, 
                                            test_data=test_data.values, 
                                            exclude_unknowns=False,
                                            verbose=True
                                        )
    
    ## Initialize a BPR model from cornac module 
    bpr = cornac.models.BPR(
                                k=k, 
                                max_iter=epochs, 
                                learning_rate=learning_rate, 
                                lambda_reg=0.001, 
                                verbose=True
                            )
    
    ## Initilize metric objects
    map_metric = cornac.metrics.MAP()       # Function to estimate MAP
    ndcg_metric = cornac.metrics.NDCG(k=k)  # Function to estimate NDCG @top K
    mrr_metric = cornac.metrics.MRR()       # Function to estimate MRR

    ## Model training and evaluation
    test_result, val_result = eval_method.evaluate(
                                                        model=bpr, 
                                                        metrics=[map_metric, ndcg_metric, mrr_metric], 
                                                        user_based=True
                                                )

    return test_result