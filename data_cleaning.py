import pandas as pd
import numpy as np

df = pd.read_csv('tags.csv')

domains = ['sci-fi','atmospheric','action','comedy','surreal','based on a book','funny','twist ending','visually appealing','dystopia','dark comedy','BD-R','romance','fantasy']

movietags = pd.read_csv('genome-tags.csv')

movies = pd.read_csv('movies.csv')

genres = [i for i in movies.genres.values if '|' not in i]
genres =list(sorted(set(genres)))

small_genres = ['Drama', 'Comedy', 'Adventure', 'Action', 'War', 'Documentary', 'Horror', 'Sci-Fi', 'Children']


def return_bools(row, small_genres):
     return [i in row.genres for i in small_genres]

bolo_list = []
for i in movies.iterrows():
    bolo_list.append(return_bools(i[1], small_genres))

bolo_list = np.array(bolo_list)
for idx,i in enumerate(small_genres):
    movies[i] = bolo_list[:, idx]
    


ratings = pd.read_csv('ratings.csv')

include_film = [movies[i] for i in small_genres]
include_film_id = []
for j, i in movies.iterrows():
    any_found = False
    for k in small_genres:
        if i[k] :
            any_found = True
        else:
            pass
    if any_found:
        include_film_id.append(j)


movies = movies.loc[include_film_id]
ratings = ratings[ratings['movieId'].isin(include_film_id) ]
tag_df = pd.read_csv('genome-scores.csv')
tag_df = tag_df[tag_df.relevance >= 0.2]
tags_to_use = []
for i, j in Counter(tag_df.tagId).most_common():
    if i >= 500:
        tags_to_use.append(j)
tag_df = pd.read_csv('genome-scores.csv')
tag_df = tag_df[tag_df.tagId.isin(tags_to_use)]
tag_df = tag_df[tag_df.movieId.isin(movies.movieId.values)]  

movies = movies[movies.movieId.isin(tag_df.movieId.values)]
tag_ids = list(sorted(set(tag_df.tagId.values)))
for i in tag_ids:
    movies['tag ' + str(i)] = tag_df[tag_df.tagId == i]['relevance'].values


movie_to_genre = {}
for idx, flick in movies.iterrows():
    bins = [flick.Drama, flick.Comedy, flick.Documentary, flick.Action]
    movie_to_genre[idx] = bins 

titles = [movies.columns[0]] + small_genres
ratings = ratings.merge(movies[titles], on='movieId',how='left')


for j in ratings.columns[4:]:
    ratings[j] = pd.to_numeric(ratings[j])


rating_averages = ratings.groupby('userId').mean()
rating_averages[small_genres].min(axis=1).to_numpy().nonzero() 
small_genres = ['Drama', 'Comedy', 'Adventure', 'Action', 'War', 'Children']

user_rats = ratings.groupby('userId').mean()
user_id_avgs = rating_averages[rating_averages[small_genres].min(axis=1) >0]
user_prefs = list(set(user_id_avgs.reset_index().userId.values))
ratings = ratings[ratings.index.isin(user_prefs)]


import pickle as pl 

with open('ratingsratsmovies.pl', 'wb') as f:
    pl.dump([ratings, user_rats, movies], f)



