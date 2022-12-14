# Recommender Systems
> #### _Archit, Shen, Shrey | Fall '22 | AIPI 590 Take Home Challenge_
&nbsp;

## About the Project

**Task:** 

Train different session (contextual, sequential) based product recommendation
recommenders for E-commerce use case and compare the performance of the recommenders.

**Requirements:**

In the deliverables and experiments, one of the recommenders needs to be a Deep RL
recommender [DRL2, DRL1, or DRL3] and at least two different datasets are used for
training/testing. Also, at least two offline evaluation metrics are used for benchmarking.

**Our Approach:**

Lorem ipsum Lorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsum

&nbsp;
## Datasets overview
We have used two E-commerce datasets for our project. Following are the details of the datasets used

**1. Dataset #1 - Retail Rocket Dataset**

The first dataset used was from [Retail Rocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Retail Rocket is a company that generates personalized product recommendations for shopping websites and provides customer segmentation based on user interests and other parameters. The dataset was collected from a real-world e-commerce website and consists of raw data, i.e. data without any content transformation. However, all values are hashed to address confidentiality concerns. Among the files in the dataset, only the behavior data (_events.csv_) is used in this project. The behavior data is a timestamped log of events like clicks, add to carts, and transactions that represent different interactions made by visitors on the e-commerce website over a time period of 4.5 months. There are a total of 2756101 events produced by 1407580 unique visitors. 

**2. Dataset #2 - H&M Dataset**

The second dataset used was from [H&M Group](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=transactions_train.csv). H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Thier online store offers shoppers an extensive selection of products to browse through. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images. Among the files in the datset, only the transactions data (_transactions_train.csv_) has been used in our project. This file consists of the purchases each customer for each date, as well as additional information like price and sales chanel. Duplicate rows correspond to multiple purchases of the same item.

&nbsp;
## Steps to run the code

We have created driver jupyter notebooks for each model type (DRL and Non DRL) and both the datasets (Retail Rocket and H&M). These notebooks can be uploaded to the Google Colab and they contain all the code and corresponsing descriptions to reproduce the results.

All the required datasets have been uploaded to an AWS S3 bucket. The driver notebooks autmatically clone our github repository and download the required datasets from the S3 bucket before executing modelling and evaluation scripts.

&nbsp;

**1. Run Deep Reinforcement Learning Models**

Jupyter notebook at  `DRL_Recommenders/Dataset_1_Retail_Rocket/RR_SA2C_Recommender.ipynb` contains all the code and corresponding descriptions to reproduce the results for Retail Rocket Dataset. Similary `DRL_Recommenders/Dataset_2_HM/HM_SNQN_Recommender.ipynb` contains all the code and corresponding descriptions to reproduce the results for H&M Dataset. For your convenience, the steps are recapped below:
1. Launch `Driver Notebook` in a Google Colab instance
2. Run the first cell to clone the git repository containing all source code
3. Run the second cell to install required Python library
4. Run the third cell to download the required dataset to the Colab instance
5. Run the fourth cell to pre-process data and generate replay buffer for Deep Reinforcement Learning
6. Run the final cell to begin model training and evaluation

&nbsp;

**1. Run Non Deep Reinforcement Learning Models**

Jupyter notebook at  `Non_DRL_Recommenders/Dataset_1_Retail_Rocket/BPR_Retail_Rocket.ipynb` contains all the code and corresponding descriptions to reproduce the results for Retail Rocket Dataset. Similary `DRL_Recommenders/Dataset_2_HM/BPR_HM.ipynb` contains all the code and corresponding descriptions to reproduce the results for H&M Dataset. For your convenience, the steps are recapped below:
1. Launch `Driver Notebook` in a Google Colab instance
2. Run the first cell to clone the git repository containing all source code
3. Run the second cell to install required Python library
4. Run the third cell to download the required dataset to the Colab instance
5. Run the fourth and following cells to pre-process data and generate training dataset
6. Run the last cell to train and evaluate the model and produce evaluation metrics

&nbsp;
## Evaluation Metrics & Results
The evaluation metrics used are Normalized Discounted Cumulative Gain (NDCG), Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and Hit Ratio (HR).

- NDCG@k measures the quality of the recommendation list based on the top-k ranking of items in the list higher ranked items are scored higher
- RR@K' ("reciprocal-rank-at-k"): is the inverse rank (one divided by the rank) of the first item among the top-K recommended that is in the test data. The average across users is typically referred to as the "Mean Reciprocal Rank" or MRR
- map@K measures the average precision@K averaged over all queries (for the entire dataset)
- HR@k measures whether the ground-truth item is in the top-k positions of the recommendation list generated by the model

&nbsp;

**1. Results for Retail Rocket Dataset**
|DRL Model Results|Non DRL Model Results|
|--|--|
|<table> <tr><th>NDCG@10</th><th>HR@10</th></tr><tr><td>0.5180</td><td>0.6250</td></tr> </table>|<table> <tr><th>NDCG@10</th><th>MRR@10</th><th>MAP@10</th></tr><tr><td>0.0007</td><td>0.0015</td><td>0.0015<td></tr> </table>|

**2. Results for H&M Dataset**
|DRL Model Results|Non DRL Model Results|
|--|--|
|<table> <tr><th>NDCG@10</th><th>HR@10</th></tr><tr><td>0.5180</td><td>0.6250</td></tr> </table>|<table> <tr><th>NDCG@10</th><th>MRR@10</th><th>MAP@10</th></tr><tr><td>0.0014</td><td>0.0026</td><td>0.0026<td></tr> </table>|

&nbsp;
# Folder structure

```
📦recommenders_aipi590
┣ 📂DRL_Recommenders
 ┃ ┣ 📂Dataset_1_Retail_Rocket
 ┃ ┃ ┣ 📂RR_data
 ┃ ┃ ┃ ┗ 📜README.md
 ┃ ┃ ┣ 📂src
 ┃ ┃ ┃ ┣ 📜NextItNetModules_v2.py
 ┃ ┃ ┃ ┣ 📜SA2C_v2.py
 ┃ ┃ ┃ ┣ 📜SASRecModules_v2.py
 ┃ ┃ ┃ ┣ 📜gen_replay_buffer.py
 ┃ ┃ ┃ ┗ 📜utility_v2.py
 ┃ ┃ ┣ 📜RR_SA2C_Recommender.ipynb
 ┃ ┃ ┗ 📜requirements.txt
 ┃ ┣ 📂Dataset_2_HM
 ┃ ┃ ┣ 📂HM_data
 ┃ ┃ ┃ ┗ 📜README.md
 ┃ ┃ ┣ 📂src
 ┃ ┃ ┃ ┣ 📜NextItNetModules_v2.py
 ┃ ┃ ┃ ┣ 📜SASRecModules_v2.py
 ┃ ┃ ┃ ┣ 📜SNQN_v2.py
 ┃ ┃ ┃ ┣ 📜gen_replay_buffer.py
 ┃ ┃ ┃ ┗ 📜utility_v2.py
 ┃ ┃ ┣ 📜HM_SNQN_Recommender.ipynb
 ┃ ┃ ┣ 📜HM_SNQN_SASRec.ipynb
 ┃ ┃ ┗ 📜requirements.txt
 ┃ ┗ 📜README.md
 ┣ 📂Non_DRL_Recommenders
 ┃ ┣ 📂Dataset_1_Retail_Rocket
 ┃ ┃ ┗ 📜BPR_Retail_Rocket.ipynb
 ┃ ┣ 📂Dataset_2_HM
 ┃ ┃ ┗ 📜BPR_HM.ipynb
 ┃ ┣ 📜README.md
 ┃ ┣ 📜bpr_model.py
 ┃ ┗ 📜requirements.txt
 ┣ 📜.gitignore
 ┗ 📜README.md
```