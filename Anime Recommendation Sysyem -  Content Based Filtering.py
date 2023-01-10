#import necesarry dependencies and packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#load anime dataset
anime_data = pd.read_csv('C:\\Users\\Ankush R. Chowdhury\\Downloads\\anime.csv', low_memory = False)
rating_data = pd.read_csv('C:\\Users\\Ankush R. Chowdhury\\Downloads\\rating\\rating.csv', low_memory = False)

#show first 5 records of anime and rating.csv
print(anime_data.head(5))
print(rating_data.head(5))

#print shape of anime_data
print ("The shape of the  data is (row, column):"+ str(anime_data.shape))

#print shape of rating_data
print ("The shape of the  data is (row, column):"+ str(rating_data.shape))

#print rating_data info
print (rating_data.info())

#making a new dataframe anime_fulldata and merging data from anime_data and rating_data using the anime_id as reference
anime_fulldata=pd.merge(anime_data,rating_data,on='anime_id',suffixes= ['', '_user'])

#renaming rating_user column in anime_fulldata as user_rating
anime_fulldata = anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})
print(anime_fulldata.head())

# Creating a dataframe for rating counts
#dropping all NA values in anime_fulldata
combine_anime_rating = anime_fulldata.dropna(axis = 0, subset = ['anime_title'])

#first we are grouping combine_anime_rating by anime_title then we are taking a count of it then we are resetting the index and then renaming rating column to totalRatingcount
anime_ratingCount = (combine_anime_rating.
     groupby(by = ['anime_title'])['user_rating'].
     count().
     reset_index().rename(columns = {'rating': 'totalRatingCount'})
    [['anime_title', 'user_rating']]
    )

#Exploratory Data Analysis
#making a new dataframe top10_animerating which comprises of anime_ratingcount with anime_title and user_rating columns
#and sorting all rows by user_rating in descending order
top10_animerating=anime_ratingCount[['anime_title', 'user_rating']].sort_values(by = 'user_rating',ascending = False).head(10)

#making a barplot of anime_title and user_rating
ax=sns.barplot(x="anime_title", y="user_rating", data=top10_animerating, palette="Dark2")

#setting extra parameters and plotting user rating count graph
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40, ha="right")
ax.set_title('Top 10 Anime based on rating counts',fontsize = 22)
ax.set_xlabel('Anime',fontsize = 20) 
ax.set_ylabel('User Rating count', fontsize = 20)
plt.show()

#updating anime_fulldata dataframe by merging anime_ratingCount dataframe to it using the reference of anime_title and updating all rows from the left direction.
anime_fulldata = anime_fulldata.merge(anime_ratingCount, left_on = 'anime_title', right_on = 'anime_title', how = 'left')

#renaming user_Rrating column to user_rating_x and totalratingcount to user_rating_y
anime_fulldata = anime_fulldata.rename(columns={'user_rating_x': 'user_rating', 'user_rating_y': 'totalratingcount'})

#making a copy of anime_fulldata and storing it in duplicate_anime dataframe
duplicate_anime=anime_fulldata.copy()
#dropping duplicate records from the dataframe
duplicate_anime.drop_duplicates(subset ="anime_title", 
                     keep = 'first', inplace = True)

#making a new dataframe top10_animemembers which consists of anime_title and members and the rows are sorted in descending order on the basis of members.
top10_animemembers=duplicate_anime[['anime_title', 'members']].sort_values(by = 'members',ascending = False).head(10)

#plotting the top10_animemembers in the form of a barplot
ax=sns.barplot(x="anime_title", y="members", data=top10_animemembers, palette="gnuplot2")

#setting extra parameters and plotting the community size graph
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40, ha="right")
ax.set_title('Top 10 Anime based on members',fontsize = 22)
ax.set_xlabel('Anime',fontsize = 20) 
ax.set_ylabel('Community Size', fontsize = 20)
plt.show()

#plotting histogram for rating of websites and rating of users
plt.figure(figsize = (15, 7))
plt.subplot(1,2,1)
anime_fulldata['rating'].hist(bins=70)
plt.title("Rating of websites")
plt.subplot(1,2,2)
anime_fulldata['user_rating'].hist(bins=70)
plt.title("Rating of users")
plt.show()

#plotting a piechart on the medium of streaming using plotly
import plotly.graph_objects as go
labels = anime_fulldata['type'].value_counts().index
values = anime_fulldata['type'].value_counts().values
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(
    title={
        'text': "Medium of Streaming",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()

#new dataframe nonull_anime which is a copy of anime_fulldata
nonull_anime=anime_fulldata.copy()

#dropping all na values on nonull_anime
nonull_anime.dropna(inplace=True)

#plotting wordcloud on anime genre
from collections import defaultdict

all_genres = defaultdict(int)

for genres in nonull_anime['genre']:
    for genre in genres.split(','):
        all_genres[genre.strip()] += 1
        
from wordcloud import WordCloud

genres_cloud = WordCloud(width=800, height=400, background_color='white', colormap='gnuplot').generate_from_frequencies(all_genres)
plt.imshow(genres_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()



anime_feature=anime_fulldata.copy()
anime_feature["user_rating"].replace({-1: np.nan}, inplace=True)
anime_feature.head()

anime_feature = anime_feature.dropna(axis = 0, how ='any') 
anime_feature.isnull().sum()

anime_feature['user_id'].value_counts()

counts = anime_feature['user_id'].value_counts()
anime_feature = anime_feature[anime_feature['user_id'].isin(counts[counts >= 200].index)]

anime_pivot=anime_feature.pivot_table(index='anime_title',columns='user_id',values='user_rating').fillna(0)
print(anime_pivot.head())

#from scipy.sparse import csr_matrix

#anime_matrix = csr_matrix(anime_pivot.values)

#from sklearn.neighbors import NearestNeighbors


#model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
#model_knn.fit(anime_matrix)

#query_index = np.random.choice(anime_pivot.shape[0])
#print(query_index)
#distances, indices = model_knn.kneighbors(anime_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

#for i in range(0, len(distances.flatten())):
#    if i == 0:
#        print('Recommendations for {0}:\n'.format(anime_pivot.index[query_index]))
#    else:
#        print('{0}: {1}, with distance of {2}:'.format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


import re
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    
    return text

anime_data['name'] = anime_data['name'].apply(text_cleaning)

from sklearn.feature_extraction.text import TfidfVectorizer


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
anime_data['genre'] = anime_data['genre'].fillna('')
genres_str = anime_data['genre'].str.split(',').astype(str)
tfv_matrix = tfv.fit_transform(genres_str)
print(tfv_matrix.shape)

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(anime_data.index, index=anime_data['name']).drop_duplicates()

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    anime_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                 'Rating': anime_data['rating'].iloc[anime_indices].values})

print(give_rec('Naruto: Shippuuden'))

print(give_rec('Steins;Gate'))

print(give_rec('Shingeki no Kyojin'))

print(give_rec('One Piece'))

print(give_rec('Dragon Ball Z'))
