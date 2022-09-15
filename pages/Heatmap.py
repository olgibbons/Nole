import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from ipywidgets import interact, interactive, fixed, interact_manual
import pathlib
base_path = pathlib.Path(__file__).resolve().parent
pic_path = '../Screenshot 2022-09-15 at 12.41.32.png'
pic_full_path = base_path.joinpath(base_path, pic_path)
csv_name = "../first_frame.csv"
csv_full_path = base_path.joinpath(base_path, csv_name)
individ_first = pd.read_csv(csv_full_path)
im1 = Image.open(pic_full_path)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    return im#, cbar





def heat_plotter(num_rows):
    ranked_first = individ_first.sort_values(by = "winner_rank")

    output = ranked_first.iloc[:num_rows,:].agg({"T_1st_ad_%ServesWon": ["mean"],
                            'M_1st_ad_%ServesWon': ["mean"],
                             'W_1st_ad_%ServesWon': ["mean"],
                             'T_1st_d_%ServesWon': ["mean"],
                             'M_1st_d_%ServesWon': ["mean"],
                            'W_1st_d_%ServesWon':["mean"]})


    rearranged = output[["W_1st_ad_%ServesWon",
                     "M_1st_ad_%ServesWon",
                     "T_1st_ad_%ServesWon",
                     "T_1st_d_%ServesWon",
                     "M_1st_d_%ServesWon",
                     "W_1st_d_%ServesWon"]]


    col_labs = rearranged.columns.tolist()

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
#     color_array = np.array([0.3]*6).reshape(1,6)
    player_serves =  np.hstack([rearranged, rearranged]).reshape(2,6)
    im = heatmap(player_serves, ["","","","","",""], rearranged.columns.tolist(), ax=ax,
                           cmap="YlGn", cbarlabel="Serves won")
    return im

def final_plotter(num_rows):
    ranked_first = individ_first.sort_values(by = "winner_rank")

    output = ranked_first.iloc[:num_rows,:].agg({"T_1st_ad_%ServesWon": ["mean"],
                                'M_1st_ad_%ServesWon': ["mean"],
                                 'W_1st_ad_%ServesWon': ["mean"],
                                 'T_1st_d_%ServesWon': ["mean"],
                                 'M_1st_d_%ServesWon': ["mean"],
                                'W_1st_d_%ServesWon':["mean"]})


    rearranged = output[["W_1st_ad_%ServesWon",
                         "M_1st_ad_%ServesWon",
                         "T_1st_ad_%ServesWon",
                         "T_1st_d_%ServesWon",
                         "M_1st_d_%ServesWon",
                         "W_1st_d_%ServesWon"]]
    arr = np.array(rearranged)
    arr_small= (arr - np.min(arr)) / (np.max(arr)-np.min(arr))

    cmap = plt.cm.get_cmap("autumn")

    colors_list = []
    for val in arr_small.tolist():
        colors_list.append(cmap(val))


    im = np.array(im1)/255

    im[120:600,140:250,:] = colors_list[0][0]

    im[120:600,260:370,:] = colors_list[0][1]

    im[120:600,380:480,:] = colors_list[0][2]

    im[120:600,490:600,:] = colors_list[0][3]

    im[120:600,610:720,:] =colors_list[0][4]

    im[120:600,725:830,:] = colors_list[0][5]
    fig = plt.figure()
    plt.imshow(im)
    plt.axis("off")
    st.pyplot(fig)

# interact(final_plotter, num_rows= (1,100,1))
n_rows = st.slider("This is the number of top players to include in the heatmap", min_value = 1, value = 50, max_value = 100)
final_plotter(n_rows)
