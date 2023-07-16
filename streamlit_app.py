import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import itertools as it
from _lib.christofides import *

st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-color: gray;
        }
    </style>""",
    unsafe_allow_html=True,
)

# plot settings
sns.reset_defaults()
sns.set(
    rc={
        'figure.figsize': (8,5),
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'black',
    }
)

def gen_data(num):
    # np.random.seed(945)
    points = np.random.sample(size=(num,2))*100
    return points

def show_data(X):
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # ax.set_facecolor("black")
    fig.patch.set_facecolor('#0E1117')
    ax.scatter(X[:,0], X[:,1], color="xkcd:gray")
    plot = st.pyplot(fig)
    return fig, plot 

def distance(X,path):
    d = 0
    curr = path[0]
    for ind in path[1:]:
        d += np.linalg.norm(X[ind]-X[curr])
        curr = ind
    return d

def show_path(X,path,fig,plot,weight=1.0,pathType="normal"):
    ax = fig.axes[0]
    curr = path[0]
    for ind in path[1:]:
        x1,y1 = X[curr]
        x2,y2 = X[ind]
        xs = [x1,x2]
        ys = [y1,y2]
        if pathType=="best":
            ax.plot(xs,ys, color="xkcd:reddish")   
        elif pathType=="approx":
            ax.plot(xs,ys, color="xkcd:gold", linewidth=.5, alpha=1.0)
        else: 
            ax.plot(xs,ys, color="xkcd:gunmetal", linewidth=weight**10, alpha=weight**10)
        curr = ind


placeholder_bar0, placeholder_bar1 = st.empty(), st.empty()
my_bar_gen = placeholder_bar0.progress(0, text="Erzeugte Pfade:")
my_bar_plot = placeholder_bar1.progress(0, text="Gezeichnete Pfade:")

st.title('Traveling Salesperson')
num  = st.slider('Anzahl der Datenpunkte', 3, 10, 6, key='num_linear')

# data
X = gen_data(num)
fig, plot = show_data(X)
plot.pyplot(fig)


## exhaustive search for best path
t0 = time.time()
path = list(range(num)) + [0]
perms = it.permutations(path[1:-1])
tot = np.math.factorial(num-1)
paths = np.zeros((tot,num+1), dtype=np.int32)
dists = np.zeros(tot)
i = 0
for p in perms:
    thispath = [0]+list(p)+[0]    
    thisdist = distance(X,thispath)
    paths[i,:] = thispath
    dists[i] = thisdist
    i += 1
    t1 = time.time()
    my_bar_gen.progress(i/tot, text="Erzeugte Pfade: {}/{} (Dauer: {} ms)".format(i,tot,round(1000*(t1-t0),1)))

bestindex = np.argmin(dists)
mindist = dists[bestindex]
bestpath = paths[bestindex]
scores = (dists/mindist)**(-1)
tf = time.time()
my_bar_gen.progress(1.0, text="Erzeugte Pfade: {}/{}  (Dauer: {} ms)".format(i,tot,round(1000*(tf-t0),1)))

## christofides approximation
tc0 = time.time()
c_len, c_path = tsp(X)
tc1 = time.time()
st.markdown("Christofides-Approximation: Distanz :orange[{}] (vs. :red[{}]), Dauer :orange[{}] ms (vs. :red[{}] ms)".format(round(c_len,2), round(mindist,2), round(1000*(tc1-tc0),1),round(1000*(tf-t0),1)))

## visualization
### lineplot (paths)
show_path(X,c_path,fig,plot,pathType="approx")
show_path(X,bestpath,fig,plot,pathType="best")
plot.pyplot(fig)

for ind in range(tot):
    if ind != bestindex:
        show_path(X,paths[ind],fig,plot,weight=scores[ind])
    my_bar_plot.progress((ind+1)/tot, text="Gezeichnete Pfade: {}/{}".format(ind+1,tot))
show_path(X,c_path,fig,plot,pathType="approx")
show_path(X,bestpath,fig,plot,pathType="best")
plot.pyplot(fig)

### barplot (distances)
fig0, ax0 = plt.subplots(figsize=(20,3))
plt.axis('off')
fig0.patch.set_facecolor('#0E1117')
barlist = ax0.bar(range(len(dists)),dists, color="xkcd:dark gray", linewidth=0)
## find entry in distances corr to christofides solution:
c_ind = np.argmin((c_len - dists)**2)
barlist[c_ind].set_color('xkcd:ochre')
barlist[bestindex].set_color('xkcd:reddish')
st.pyplot(fig0)





