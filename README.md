# Recommendation_System_movielense
## Collaborative Filtering Recommendation System for Movie lens dataset using the Scikit-learn Surprise library

A recommendation engine filters the data using different machine learning algorithms to recommend the most relevant products or items to a particular user based on past behaviors. It captures the past behaviors of customers which can be collected implicitly or explicitly and operates on the principle of finding patterns of the data to recommend the product users might like. (Sharma, 2018)
There are three main types of recommendation engines and they are,

1)	Content-based filtering recommendation engine
2)	Collaborative filtering recommendation engine
3)	Hybrid model recommendation engine

## Content-based filtering

Content-based filtering recommendation algorithm uses a profile of customer preferences and a description of products and products are described using keywords such as genre, color, product type, content, etc which are implicit data. This works on the principle that if the user likes a particular item, then the user will like similar other items. To check the similarity between items it uses the cosine similarity and the Euclidean distance. (Sharma, 2018)

Since this algorithm is based on the products that the users like previously, this recommendation system is limited only to the products which are already used. Thus recommend only the products which are similar to the already using products. 

## Collaborative filtering

Collaborative filtering recommendation algorithm is based on collecting and analyzing data on user behavior, preferences, activities and recommending what a user might like based on the similarity with other users. (Techlabs)

This collaborative filtering recommendation system does not require any additional information (such as the contents of the product) to analyze the data because it is capable of accurately recommending complex items without requiring an "understanding" of the item itself. And also it requires only the explicit rating. They are two main techniques in collaborative filtering and they are,

	Memory-Based Collaborative filtering
	Model-Based Collaborative filtering

Memory Based collaborative filtering approach is focusing on computing the relationships between items or users. Here evaluates a user’s preference for an item, based on the similarities among neighbors in memory. There are two main techniques in the memory-based collaborative filtering recommendation technique.  

	User-based collaborative filtering
	Item-based collaborative filtering

Both methods need to identify the N-Nearest Neighbors to the target sample, be this one a user or an item. (Claudio, 2014)

The model-Based collaborative filtering approach derives a model and uses it to compute the recommendation. Generally, the Model-Based approach produces a model offline, while the recommendation takes place online, and is virtually instantaneous. (Claudio, 2014) The most popular approaches in this technique are based on low-dimensional factor models which are model-based matrix factorization models.
Moreover, there is a problem in collaborative filtering called the ‘cold-start problem’, which describes the difficulty of making recommendations when the users or the items are new. But the considering datasets in the analysis are free from that problem because to be a customer of the company, customers must have purchased at least one connection with the company. 

### User Based Collaborative Filtering

This algorithm first focuses on the similarity scores between users. Based on the computed similarity scores, the algorithm identifies the most similar users and recommends products according to the previously liked or bought products.  (Sharma, 2018)

The principle behind this algorithm is as follows, if both persons A and B like ‘Movie 1’ and ‘Movie 2’, then they have similar interests. If B likes to ‘Movie 3’ then it is highly likely to A also would like for ‘Movie 3’. 

![user based](https://github.com/Sehaniw0802/Recommendation_System_movielense/assets/66731646/8d25bed6-29f9-4c7f-aa79-826bdb11af65)

Figure: User-Based Collaborative Filtering

To calculate the similarity scores this is using the weighted sum of the user ratings given by other users to the item (i), and to predict the ratings for users following steps are following. (Sharma, 2018)

	First, need to calculate the similarity between users A and B, for that focuses on the items which are rated by both users, and based on the ratings, the correlation between the users is calculated.
	Once calculated the similarity scores between each user, then predictions are calculated based on those calculated similarity values. 
	Then based on those calculated predictions values recommendations are made.

This algorithm is time-consuming since it calculates the similarity score for each user and calculates predictions for each similarity score.  So to handle this problem grouping the users into homogeneous clusters is one of the solutions.  

### Item Based Collaborative Filtering

This Item-based technique focuses on the similarities between items. These similarities are calculated by the ratings given by the users to each item. For that in this method, analyzing the user-item matrix to identify relationships between different items.  Then based on these relationships, indirectly computing the recommendations for users. (Badrul Sarwar)

The principle behind this algorithm is as follows.

In figure 2.13, ‘Item 1’ and ‘Item 3’ are similar to each other as they have been liked by ‘User 1’ and ‘User 2’. To recommend items to ‘User 3’, first, need to find the items being liked by ‘User 3’. (‘Item 3’). The next step is to find similar items to ‘Item 3’. (‘Item 1’). Therefore, can recommend ‘Item 1’ to ‘User 3’ as he likes ‘Item 3’ and ‘Item 3’ is similar to ‘Item 1’.

![item based](https://github.com/Sehaniw0802/Recommendation_System_movielense/assets/66731646/ecde6631-63d2-4c78-8c7b-4f34c9387bf5)
                                
Figure: Item-based Collaborative Filtering

The steps of this method can be summarized as below (rachitgupta, 2020). 

	Build the model by calculating similarity scores for all the item pairs by using similarity measures (ex; cosine similarity, Pearson similarity).
	Execute a recommendation system by using the already rated items that are most similar to the missing item to generate the rating. Compute this using a formula that computes the rating for a particular item using the weighted sum of the ratings of the other similar products.

## Used Similarity Measures

### Cosine Similarity

The Cosine Distance similarity function focuses on the two samples to compare and consider the angle between them. To quantify the similarity among them the cosine of the angle is considered. The scale used in this similarity measure is from +1 to -1 and positive values near to +1 represent high similarity. The negative values near to minus 1 represent the inversely high correlation (when one says True the other says False). and zero represents no correlation. (Claudio, 2014)

![cosine similarity](https://github.com/Sehaniw0802/Recommendation_System_movielense/assets/66731646/5d6c527d-aedc-4888-9d77-82f17a29fbb9)
           
Figure: Cosine Similarity

The cosine similarity uses cos(θ) to measure the distance between two vectors. So when θ increases, cos(θ) decreases (cos(θ) = 1 when θ = 0 and cos(θ) = 0 when θ = 90). Therefore, when the value of θ is smaller, the two vectors are considered closer which means high similarity. 

### Pearson Correlation Similarity

The Pearson Correlation similarity function is one of the best-known algorithms for Collaborative filtering Recommender Systems. This uses as the baseline for comparisons. The similarities of the Pearson Correlation are represented by a scale of -1 to +1 same as the cosine similarity. Here positive high value suggests a high correlation and a negative high value suggests an inversely high correlation. The correlation zero indicates that considering users or items are not correlated. (Claudio, 2014)

### Mean Squared Difference Similarity

Mean Squared Difference computes the difference between ratings. Here compute the Mean Squared Difference similarity between all pairs of users (or items) and only common users (or items) taken into account.

## Matrix Factorization Method 

Matrix Factorization is one of the most popular approaches of collaborative filtering and it is based on latent factor models. By this method can find some latent features that can determine how a user rates an item. This method decomposes the matrix into constituent parts in a way such that the dot product of these matrices reproduces the best approximation of the original matrix. 

![matrix factorization method](https://github.com/Sehaniw0802/Recommendation_System_movielense/assets/66731646/09c9f674-f652-4532-9c5b-e5c3cf3aac6d)
 
Figure: Matrix Factorization method

Some of the most successful latent factor models are based on matrix factorization. In its natural form, matrix factorization characterizes items and users using vectors of factors inferred from item rating patterns. High correspondence between item and user factors leads. (Filtering, 2017)

Let R n*m be a rating matrix containing the ratings of ‘n’ users for ‘m’ items. Each matrix element refers to the rating of user u for item i. Given a lower dimension d, the Matrix factorization technique factorizes the raw matrix R n*m into two latent factor matrices. One is the user-factor matrix P n*d and the other is the item-factor matrix Q d*m. The factorization is done in a way such that R is approximated as the inner product P and Q.

### Singular Value Decomposition (SVD)

Singular value decomposition (SVD) is one of the common matrix factorization methods used in collaborative filtering that generalizes the eigen decomposition of a square matrix (n x n) to any matrix (n x m) where m< n. So this is generally used as a dimensionality reduction technique in machine learning. In recommendation engines when baselines are not used, this is equivalent to Probabilistic Matrix Factorization.  (Flynn, 2021)

### Non-negative Matrix Factorization (NMF)

Nonnegative matrix factorization is a matrix decomposition approach that decomposes a nonnegative matrix into two low-rank matrices constrained to have nonnegative elements. This results in a reduced representation of the original data that can be seen either as feature extraction or as a dimensionality reduction technique. This algorithm is very similar to the SVD algorithm and user and item factors are kept positive in this algorithm. (Nazir, 2021)

## Accuracy Measures of the Recommendation engine

### Root Mean Square Error (RMSE)

Root Mean Square Error (RMSE) is a standard way of measuring the error of a model in predicting quantitative data. It shows how far predictions fall from measured true values using Euclidean distance. It can be defined as below. (Hug, 2015)

![RMSE](https://github.com/Sehaniw0802/Recommendation_System_movielense/assets/66731646/5d38de31-09d5-4c51-98bf-7575e9bf8df3)
   
Here rui represents the ‘true’ rating of user u for item i
r^ui represents the ‘estimated’ rating for user u for item i
R^ represents the set of predicted ratings.

### Precision

Precision is one indicator of checking the performance of machine learning models. It refers to the number of true positives divided by the total number of positive predictions. Here the total number of positive predictions is equal to the summation of the number of true positives and the number of false positives. (Machine Learning) In this collaborative filtering technique precision is defined as below. (Hug, 2015)

Precision=( |{Recommended items that are relevant}|)/(|{Recommended items}|)

In this algorithm, an item is considered as ‘relevant’ if its true rating is greater than the defined threshold value. An item is considered as ‘recommended’ if its estimated rating is greater than the defined threshold value and that rating belongs to the highest k estimated ratings. And also if division by zero occurs and then precision becomes undefined and as a convention set it into 0.

### Recall

The recall is the fraction of positives that are correctly classified. The recall is also commonly referred to as ‘true positive rate’ and ‘sensitivity’. In Collaborative Filtering algorithms recall is defined as below. (Hug, 2015)

Recall=(|{Recommended items that are relevant}|)/(|{Relevant items}|)

Here also terms ‘relevant’ and ‘recommended’ are defined the same as the precision and if division by zero occurs then recall becomes undefined and as a convention set it into 0.

## Detail overview of the recommendation methods that used in this project

<img src="https://github.com/Sehaniw0802/Recommendation_System_movielense/assets/66731646/52ab8844-6b70-4a70-8fd5-1ee1b1910be6" width="60%" height="60%"> 








