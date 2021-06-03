import sys
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.random.seed(1000)

# load embeddings
input_path = sys.argv[1]  # pickle file that contains the embeddings


with open(input_path, 'rb') as f:
    data = pickle.load(f)

number_of_instance = -1

relation_vectors = data["relation_mat"]  # R+1, D
instance_vectors = data["vectors"][:number_of_instance]  # N, D
predictions = data["preds"][:number_of_instance]  # N, R
labels = data["labels"][:number_of_instance]  # N, R
# relation_names = data["label_name"]  # list, len=R
# relation_names = relation_names + ["NA"]


N = instance_vectors.shape[0]
D = instance_vectors.shape[1]
R = predictions.shape[1]

# # re-center relation vectors
# relation_vectors = np.zeros((R, D))
# for i in range(R):
#     count = 0
#     vector = np.zeros(D)
#     for j in range(N):
#         if labels[j, i] == 1:
#             count += 1
#             vector = vector + instance_vectors[j]
#     vector = vector / float(count)
#     vector = vector / np.linalg.norm(vector, ord=2)
#     relation_vectors[i] = vector


# project to 2D
all_vectors = np.concatenate(
    [instance_vectors, relation_vectors], axis=0)  # N + (R+1), D
print("running tsne")
sys.stdout.flush()
all_vectors_2d = TSNE(n_components=2, verbose=1,
                      metric="cosine").fit_transform(all_vectors)

instance_vectors_2d = all_vectors_2d[:N]  # N, 2
relation_vectors_2d = all_vectors_2d[N:]  # R+1, 2
print("tsne done")
#print(instance_vectors_2d.shape, relation_vectors_2d.shape)
sys.stdout.flush()

# plot

COLORS = ["#d3d3d3",  # Light Gray
          "#ffa500",  # Orange
          "#008000",  # Green
          "#4169e1",  # Royal Blue
          "#ff0000",  # Red
          ]
# "#008000",  # Green


def draw_scatter(relation_vectors, vectors, labels, label_names=None, make_it_grey=-1, star_it=-1, cross_it=-1, name="", color_table=None):
    # relation_vectors: (R+1, 2), all relation vectors
    # vectors: (N, 2), all the vectors to plot
    # labels: (N,), label for each vector, should contain a nonnegative integer.
    # label_names: label names for legends
    # make_it_grey: indicate which label to mask out as grey color.
    # star_it: make the scatter shape of the corresponding label to be a star
    # cross_it: make the scatter shape of the corresponding label to be a cross

    plt.gcf().clear()
    scale_ = 1.0
    new_size = (scale_ * 10, scale_ * 10)
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(new_size)

    # scatter instance vectors
    viridis = plt.get_cmap('gist_ncar')
    NUM_COLORS = int(labels.max())
    #NUM_COLORS = 42
    color_table_from_cm = []
    for i in range(NUM_COLORS):
        color_table_from_cm.append(np.array([viridis(i/NUM_COLORS)]))

    if label_names != None:
        count_label = [0] * len(label_names)

    for i in range(len(vectors)):
        if labels[i] >= NUM_COLORS and labels[i] != make_it_grey:
            continue

        if labels[i] == cross_it:
            marker = 'x'
        elif labels[i] == star_it:
            marker = '*'
        else:
            marker = '.'
        x = vectors[i, 0]
        y = vectors[i, 1]
        if labels[i] == make_it_grey:
            c = '#d3d3d3'
            alpha = 0.5
        else:
            if color_table == None:
                c = color_table_from_cm[labels[i]]
            else:
                c = color_table[labels[i]]
            alpha = 0.5

        if label_names != None:
            ax.scatter(x, y, c=c, s=10, linewidths=1, marker=marker, alpha=alpha,
                       label=label_names[int(labels[i])] if count_label[int(labels[i])] == 0 else "")
            count_label[int(labels[i])] += 1
        else:
            ax.scatter(x, y, c=c, s=10, linewidths=1, marker=marker, alpha=alpha,
                       label="Instance" if i == 0 else "")

    # scatter relation vectors
    viridis = plt.get_cmap('gist_ncar')
    NUM_COLORS = int(R)
    #NUM_COLORS = 30
    color_table_from_cm = []
    for i in range(NUM_COLORS):
        color_table_from_cm.append(np.array([viridis(i/NUM_COLORS)]))

    for i in range(len(relation_vectors)):

        if i >= NUM_COLORS and i != R:
            continue

        x = relation_vectors[i, 0]
        y = relation_vectors[i, 1]
        alpha = 1.0
        if i == R:
            c = '#000000'  # black
            marker = 'v'
        else:
            c = color_table_from_cm[i]
            marker = '^'

        if i == 0:
            ax.scatter(x, y, c=c, s=20, linewidths=1, marker=marker, alpha=alpha,
                       label="Relation")
        elif i == R:
            continue
            ax.scatter(x, y, c=c, s=20, linewidths=1, marker=marker, alpha=alpha,
                       label="No Relation")
        else:
            ax.scatter(x, y, c=c, s=20, linewidths=1,
                       marker=marker, alpha=alpha, label="")

    # if label_names != None:
    ax.legend(loc='lower right', fontsize=15, numpoints=1)

    plt.title(f"TSNE: {name}", fontsize=25)
    plt.axis('off')
    # ax.set_title(f"2D plot", color='k', rotation=0,x=1.02,y=0.93, fontsize=10)

    plt.tight_layout()
    #plt.savefig('tsne.' + name + '.pdf', format='pdf', dpi=300)
    plt.savefig('tsne.' + name + '.png', format='png', dpi=300)
    plt.close()


# plot TP, TN, FP, FN of predicting not_NA

predictions_not_NA = (predictions.sum(1) > 0).astype(int)  # (N, )
label_not_nA = (labels.sum(1) > 0).astype(int)  # (N, )


true_negative = (
    (1 - predictions_not_NA) * (1 - label_not_nA)).astype(int) * 1
false_negative = (
    (1 - predictions_not_NA) * label_not_nA).astype(int) * 2
false_positive = (
    predictions_not_NA * (1 - label_not_nA)).astype(int) * 3
true_positive = (
    predictions_not_NA * label_not_nA).astype(int) * 4


draw_scatter(relation_vectors_2d, instance_vectors_2d, true_positive + true_negative + false_positive +
             false_negative, label_names=["", "TN", "FN", "FP", "TP"], make_it_grey=1, cross_it=1, star_it=4, name="Binary RETACRED(DEV)", color_table=COLORS)


# plot TP, TN, FP, FN of predicting not_NA

NA_labels = (labels.sum(1) == 0).astype(int)[:, None]  # (N, 1)
# NA_labels = np.where(NA_ids, np.ones(
#     N), np.zeros(N))[:, None]  # (N, 1)
labels_plus_NA = np.concatenate([labels, NA_labels], axis=1)  # (N, R + 1)
labels_plus_NA = np.argwhere(labels_plus_NA.astype(int) == 1)  # (N, 2)
labels_plus_NA = labels_plus_NA[:, 1]  # (N, )


draw_scatter(relation_vectors_2d, instance_vectors_2d, labels_plus_NA, make_it_grey=R,
             cross_it=R, name="All-Category RETACRED(DEV)")
