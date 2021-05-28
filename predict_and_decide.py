import pickle as pl 
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
from functools import partial
from submodular import ContinuousOptimizer
import torch.nn as nn
import random
from random import choices

import torch
from torch import nn, optim
from torch.autograd import Variable

class basic_classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
#         if torch.cuda.is_available():
#             model.cuda()
        
    def forward(self, x):
        return self.linear(x)

class better_classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 200)
#         if torch.cuda.is_available():
#             model.cuda()
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(200, 50)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(50, 20)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(20, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class hashimoto_loss(nn.Module):
    def __init__(self, eta):
        super(hashimoto_loss, self).__init__()
        self.eta = eta
        self.nonlinearity = nn.LogSoftmax()
        self.floor = nn.ReLU()
        
    def forward(self,input, target):
        loss = nn.functional.cross_entropy(input, target)
        dual_opt = self.floor(loss - self.eta)**2
        return torch.mean(dual_opt) + self.eta**2



with open('ratingsratsmovies.pl', 'rb') as f:
    ratings, user_rats, movies=pl.load(f)

batch_size = 30
iterations = 50
input_dim = 285
output_dim = 2
cutoff = 4.0
# erm_clf = basic_classifier(input_dim, output_dim)
erm_clf = better_classifier(input_dim, output_dim)

erm_loss = nn.CrossEntropyLoss()
erm_opt = optim.Adam(erm_clf.parameters(),  weight_decay=1e-5)

scores = []
for idx, flick in movies.iterrows():
    scores.append(np.mean(ratings[ratings.movieId == flick.movieId].rating.values))

movies['score overall'] = scores
del scores
movies = movies.dropna()



# for each fan, label their favorite genre

user_to_favorite_dict = {}
small_genres = ['Drama', 'Comedy', 'Adventure', 'Action', 'War', 'Children']

for user in list(set(ratings.userId)):
    current_best = ''
    current_best_score = -1
    for genre in small_genres:
        prefs = ratings[(ratings.userId == user)]
        prefs =  prefs[prefs[genre].values.astype(bool)].dropna()
        prefs = np.median(prefs)
        if prefs > current_best_score:
            current_best = genre
            current_best_score = prefs
    if current_best != '':
        user_to_favorite_dict[user] = current_best


favorite_to_user_dict = {i:[] for i in small_genres}
for i, j in user_to_favorite_dict.items():
    favorite_to_user_dict[j].append(i)

genre_scores = {}

for genre in small_genres:
    print(genre)
    genre_scores[genre] = []
    for idx, flick in movies.iterrows():
        flick_ratings = ratings[ratings.movieId == flick.movieId]
        flick_ratings = flick_ratings[flick_ratings.userId.isin(favorite_to_user_dict[genre])]

        genre_scores[genre].append(np.mean(flick_ratings.rating.values))


movies2 = deepcopy(movies)


def are_there_good_films(movies_df, ids_to_check, genres_to_check, score=4):
    relevant_movies = movies_df[movies_df.movieId.isin(ids_to_check)]
    happy_genre = [np.max(relevant_movies[genre].values) >= score for genre in genres_to_check]
    return happy_genre


indices = movies.movieId.values.tolist()
random.shuffle(indices)
cutoff = 4
train_idxs = indices[0:int(0.8*len(indices))]

test_idxs = indices[int(0.8*len(indices)):]
movie_index_to_array_row_map = {}

row_idx = 0
for i in train_idxs:
    movie_index_to_array_row_map[i] = row_idx
    row_idx += 1
for i in test_idxs:
    movie_index_to_array_row_map[i] = row_idx
    row_idx += 1
column_headers = [i for i in movies.columns if 'tag' in i]

x_train  = movies[movies.movieId.isin(train_idxs)][column_headers].values
y_train = movies[movies.movieId.isin(train_idxs)]['score overall'].values > cutoff
y_train = y_train.astype('int')
x_test = movies.loc[movies.movieId.isin(test_idxs)][column_headers].values
y_test = movies[movies.movieId.isin(test_idxs)]['score overall'].values > cutoff
y_test = y_test.astype('int')

film_array = np.concatenate([x_train, x_test])


batch_size = 30
iterations = 50
input_dim = 285
output_dim = 2

erm_clf = better_classifier(input_dim, output_dim)

erm_loss = nn.CrossEntropyLoss()
erm_opt = optim.ASGD(erm_clf.parameters(),  weight_decay=1e-5)

for epoch in range(iterations):
  for i in range(200):
    idxs = np.random.choice(len(train_idxs), size=batch_size)
    batch = x_train[idxs, :]
    erm_opt.zero_grad()
    outputs = erm_clf(torch.from_numpy(batch).float())
    loss = erm_loss(outputs, torch.from_numpy(y_train[idxs]))
    loss.backward()
    erm_opt.step()
  print(epoch, loss)

n = .1
dro_loss = hashimoto_loss(n)
# dro_model = basic_classifier(input_dim, output_dim)
dro_model = better_classifier(input_dim, output_dim)

# mle_loss = nn.CrossEntropyLoss()
dro_opt = optim.ASGD(dro_model.parameters(),  weight_decay=1e-5)
old_param = []
for epoch in range(15):
  for i in range(50):
    idxs = np.random.choice(len(train_idxs), size=batch_size)
    batch = x_train[idxs, :]
    dro_opt.zero_grad()
    outputs = dro_model(torch.tensor(batch, requires_grad=False).float())
    dro_output = dro_loss(outputs, torch.tensor(y_train[idxs], requires_grad=False) )
    dro_output.backward()
    dro_opt.step()
  print(epoch, dro_output)



  