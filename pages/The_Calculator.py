from curses.ascii import alt
from urllib.error import URLError
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
from css_styling import get_css
sys.path.append('/home/gibbons/code/TomislavMatus/Nole/pages/')
from scripts.model import *
from scripts.utils import *

#STYLING
st.markdown(get_css(), unsafe_allow_html=True)

# original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Original image</p>'
# st.markdown(original_title, unsafe_allow_html=True)

# new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">New image</p>'
# st.markdown(new_title, unsafe_allow_html=True)

#Height
number = st.number_input('Please enter your height in cm',value=185)
columns = st.columns(2)
#Imperial to metric converter
# ft = columns[0].number_input(label='Or, alternatively, Insert your height in feet',value=1, \
#     min_value=0, max_value=50, step=1)
# inch = columns[1].number_input(label='and inches',value=1, \
#     min_value=0, max_value=50, step=1)

# height = 170
# if number > 0:
#     height = number
#     st.write(f'Your height in cm is {number} cm')
# else:
#     heght = round((ft * 30.48) + (inch * 2.54))
#     st.write(f'Your Height in cm is  {round((ft * 30.48) + (inch * 2.54))} cm')

#Age
age = st.number_input('Please enter your age in years',value=40, \
    min_value=0, max_value=9001, step=1)
st.write(f"Your age, in seconds, is {age*365*24*60*60} s. (In case you're interested.)")

#Surface
surface_options = ['Hard','Clay','Carpet','Grass']
surface = st.selectbox(label='Please pick a surface', options=surface_options,\
     index=2,\
)
#OHE - 3 =0, 5 = 1
best_of_options = [3,5]
best_of = st.selectbox(label='Please pick the maximum number of sets',\
     options=best_of_options,\
     index=1\
)
min_time =0
if best_of == 3:
    min_time=18
else:
    min_time = 36
#Handedness
#left-hand is 1 and rh is 0
hand_options = ['Left','Right']
hand = st.selectbox(label='Left or Right handed?',\
     options=hand_options,\
     index=0\
)
#Aces percentage
st.markdown('Ace Percentage', unsafe_allow_html=True)
aces = st.slider(label= "", value=15.5, \
    min_value=0.0, max_value=100.0, step=0.001)

first_in = st.slider("Percentage of first serves in/ Total service points" ,value=61.2, \
    min_value=0.0, max_value=100.0, step=0.001)

first_won = st.slider("Percentage of first serves won / total first serves played" ,value=73.3, \
    min_value=0.0, max_value=100.0, step=0.001)

second_won = st.slider("Percentage of second serves won / Total second serves played" ,value=51.2, \
    min_value=0.0, max_value=100.0, step=0.001)


breaks_save = st.slider("Percentage of Break points saved /Break points faced" ,value=61.2, \
    min_value=0.0, max_value=100.0, step=1.0)

#Similarity
similarity = st.slider("Similarity to Djokovic - 1 = Djokovic, 0 = Not Djokovic" ,value=0.9, \
    min_value=0.0, max_value=1.0, step=0.01)


#Note, has been scaled on user_input and would be nice to scale on X_train instead

pred_df = pd.DataFrame()

#pred_df = surface_to_pred(surface, pred_df)
if surface == 'Hard':
    pred_df.loc[0, 'Hard'] = 1
else:
    pred_df.loc[0, 'Hard'] = 0

if surface == 'Clay':
    pred_df.loc[0, 'Clay'] = 1
else:
    pred_df.loc[0, 'Clay'] = 0

if surface == 'Carpet':
    pred_df.loc[0, 'Carpet'] = 1
else:
    pred_df.loc[0, 'Carpet'] = 0

if surface == 'Grass':
    pred_df.loc[0, 'Grass'] = 1
else:
    pred_df.loc[0, 'Grass'] = 0

#best of encoding
if best_of == 3:
    pred_df.loc[0, 'best_of'] = 0
else:
    pred_df.loc[0, 'best_of'] = 1

#left/right hand encoding
if hand == 'Left':
    pred_df.loc[0, 'opp_hand'] = 1
else:
    pred_df.loc[0, 'opp_hand'] = 0

pred_df.loc[0, 'opp_ht'] = number
pred_df.loc[0, 'opp_age'] = age
pred_df.loc[0,'Similarity'] = similarity
pred_df.loc[0, 'Aces_perc'] = aces
pred_df.loc[0,'First_serves_in_perc'] = first_in
pred_df.loc[0, 'First_serve_won_perc'] = first_won
pred_df.loc[0,'2nd_serve_win_perc'] = second_won
pred_df.loc[0,'breakpoints_saved_perc'] = breaks_save


# ['Hard', 'Clay', 'Carpet', 'Grass', 'best_of', 'opp_hand', 'opp_ht',
#        'opp_age', 'Similarity', 'Aces_perc', 'First_serves_in_perc',
#        'First_serve_won_perc', '2nd_serve_win_perc', 'breakpoints_saved_perc']
#SCALING

X_test = pred_df.to_numpy()
scaler = load_scaler("scaler_final.pickle")
X_test_scaled = scaler.transform(X_test)
#Prediction
model = load_model('new_model_perc.pickle')
pred = model.predict_proba(X_test_scaled)

chance = round(100 - pred[0][1]*100, 2)

with st.sidebar.container():
    text = f'''Chance of beating Novak:'''
    st.markdown(f'''<h1 class="stats">{text}</h1>''', unsafe_allow_html=True)
    st.metric('', value=f'{chance} %', delta=f'{round((chance-17),2)}% above average')
#TO DO LIST:
#Turn DF into np array
#preprocess np array
#feed to prediction
#display prediction

#nice to have functionised code where possible (i.e. the if elses)
#also make look pretty
