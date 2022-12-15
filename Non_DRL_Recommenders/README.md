# Non Deep Reinforcement Learning Recommenders
> #### _Archit, Shen, Shrey | Fall '22 | AIPI 590 Take Home Challange_
&nbsp;

## Bayesian Personalised Ranking (BPR)
Historically product recommender systems for e-commerce data were built using explicit ratings given by users to the products based on their opinions. However, presently there are even more advanced ways to build product recommenders that use implicit feedback instead of explicit feedback.

Implicit data is the feedback collected from customers through clicks, purchases, the number of views, add-to-cart activity, etc. Implicit feedback is more beneficial than explicit feedback as it avoids negative hate reviews from customers. It is noisy, but the vast volume compensates for that fact. It also shows more confidence in summarizing customers' likes and dislikes than in explicit feedback, which could be biased.

Bayesian Personalized Ranking optimization criterion considers user-item pairs to create more personalized rankings for each user. The primary task of personalized ranking is to provide a user with a ranked list of items. Optimization is performed based on the rank of these user-item pairs instead of scoring just on the user-item interaction. The available observations are only positive feedback, where the non-observed ones are a mixture of real negative feedback and missing values

&nbsp;
## Data preparation and transformation
In real-world datasets, users may interact with items like clicks, visits, buy, add to cart, etc. In addition, the same types of interactions may appear more than once in history. Therefore, to prepare the dataset for the BPR model, we did the following:

In the "explicit feedback" scenario, interactions between users and items are numerical/ordinal ratings or binary preferences such as likes or dislikes. But for implicit feedback cases, all we have is the interaction between users in terms of clicks, visits, and buy history, and there are no explicit ratings.

Many collaborative filtering algorithms are built on a user-item sparse matrix. We can make this matrix by aggregating our user and item interactions in many different ways:
- Count: The most simple technique is to count times of interactions between user and item for producing affinity scores
- Weighted Count: Different interactions have different weights in the count aggregation with more important interaction like buy having higher weights as compared to interations like click or add
- Negative sampling: This is based on assumptions that user-item interactions can be interpreted as preferences by taking the factors like "number of interation times", "weights", "time decay", etc. The original dataset with implicit interaction records can be binarized into one that has only 1 or 0, indicating if a user has interacted with an item, respectively.

We have personally used the "Negative sampling" technique as it helps sample the negative feedback. In this case, we can regard the items that a user has not interacted as those that the user does not like. All the existing interactions between user and item are labelled as a positive class(1) and the rest of the interactions are being labelled as negative class(0). This implies that if our model fits the training data exactly, it is going to treat all the interactions that are not present in the training data in the same manner as all of them are labelled as 0

&nbsp;
## Model
Once the implicit feeback ranking data is prepared, we split the dataset into train (80%) and test(20%) to train our BPR model. The implementation of the BPR model is from Cornac, which is a framework for recommender systems with a focus on models leveraging auxiliary data. We have writtern a script `bpr_model.py` which implements the model training, validation, and testing part on provided dataset and value of K (top items to recommend to a user). To evaluate our model on test data we have used metrics like NDCG, MRR and MAP.

**1. To Train and evaluate Retial Rocket Dataset:**
- Launch `Dataset_1_Retail_Rocket/BPR_Retail_Rocket.ipynb` in a Google Colab instance
- Run the entire notebook to prepare data, train model and generate metric results on test data

**2. To Train and evaluate H&M Dataset:**
- Launch `Dataset_2_Retail_Rocket/BPR_HM.ipynb` in a Google Colab instance
- Run the entire notebook to prepare data, train model and generate metric results on test data

&nbsp;
## Results
The evaluation metrics used are Normalized Discounted Cumulative Gain (NDCG), Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP)

- NDCG@k measures the quality of the recommendation list based on the top-k ranking of items in the list higher ranked items are scored higher.
- RR@K' ("reciprocal-rank-at-k"): is the inverse rank (one divided by the rank) of the first item among the top-K recommended that is in the test data. The average across users is typically referred to as the "Mean Reciprocal Rank" or MRR.
- The map@K measures the average precision@K averaged over all queries (for the entire dataset).

&nbsp;

**1. Results for Retail Rocket Dataset**
|Metric@k|NDCG@10|MRR@10|MAP@10|
|--|--|--|--|
|Value|0.0950|0.0015|0.0015|

**2. Results for H&M Dataset**
|Metric@k|NDCG@10|MRR@10|MAP@10|
|--|--|--|--|
|Value|0.1018|0.0026|0.0026|

&nbsp;
# References

1. Data Preparation for Colborative Filtering | Microsoft
https://github.com/microsoft/recommenders/blob/main/examples/01_prepare_data/data_transform.ipynb

2. Cornac Movie Recommendation using BPR | Microsoft
https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb

3. Bayesian Personalised Ranking (BPR) Evaluation Example | PreferredAI, Cornac
https://github.com/PreferredAI/cornac/blob/master/examples/bpr_netflix.py
https://cornac.preferred.ai/

4. BPR: Bayesian personalized ranking from implicit feedback | Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009, June).
https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf