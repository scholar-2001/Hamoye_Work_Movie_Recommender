import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def get_popular_recommendations(title, linear_sim, df):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[title]

    sim_scores = list(enumerate(linear_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies_indices = [i[0] for i in sim_scores[1:31]]
    top_movies = df[['title','popularity_score']].iloc[top_movies_indices]
    top_movies = list(top_movies.sort_values('popularity_score',ascending = False).head(5)['title'])
    top_movies_posters = [fetch_poster(mapping[title]) for title in top_movies ]
    return top_movies, top_movies_posters

st.header('Movie Recommender System')
movies = pickle.load(open('df_popularity.pkl','rb'))
df_popularity = pd.DataFrame(movies)
tfidf_matrix = pickle.load(open('tfidf.pkl','rb'))
linear_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
mapping = pickle.load(open('mapping.pkl','rb'))
movie_list = df_popularity['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = get_popular_recommendations(selected_movie ,linear_sim ,df_popularity)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

