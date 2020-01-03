import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Compute Normalized Levenshtein Distance Distribution (Cumulative and Histogram)

with open("dict_transfer_final.pkl",'rb') as f:
    data = pickle.load(f)

data = list(data.values())

fig, axs = plt.subplots(2, 1, constrained_layout=True)
fig.suptitle("Normalized Levenshtein Distance for Washington Database\
    \nComputed over the validation set (10%) of a cross-validation\
    \n TL with the full RNN and FC trainable")

sns.distplot(data,kde=False,hist_kws=dict(cumulative=True),ax=axs[0])
axs[0].set_title("Cumulative Distribution")
axs[0].set_xlabel("Normalized Levenshtein Distance")

sns.distplot(data,kde=False,ax=axs[1])
axs[1].set_title("Distribution")
axs[1].set_xlabel("Normalized Levenshtein Distance")

plt.show()