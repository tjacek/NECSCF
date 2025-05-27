import matplotlib.pyplot as plt
import scipy.stats

def scatter_plot(points,
                 title,
                 clf_x, 
                 clf_y,
                 out_path=None):
    x,y=points[:,0], points[:,1]
    pearson=scipy.stats.pearsonr(x, y) 
    text=f"corelation:{pearson.correlation:.4f},pvalue:{pearson.pvalue:.4f}"
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)
    plt.title(title)
    plt.xlabel(clf_x)
    plt.ylabel(clf_y)
    ax.annotate(text,
                xy = (0.7, -0.15),
                xycoords='axes fraction',
                ha='right',
                va="center")
    fig.tight_layout()
    plt.show()
    if(out_path):
        fig.savefig(f'{out_path}.png')

def plot_series(series_dict,
                title="Scatter",
                x_label='x',
                y_label='y',
                plt_limts=None):
    labels=['r','g','b']
    plt.figure()
    plt.title(title)
    for i,(_,points_i) in enumerate(series_dict.items()):
        for name_j,point_j in points_i:
            plt.text(point_j[0], 
                    point_j[1], 
                    name_j,
                    color=labels[i],
                    fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if(plt_limts):
        plt.xlim(plt_limts[0])
        plt.ylim(plt_limts[1])
    plt.show()

def xy_plot(conf):   
    x=utils.read_json(conf["x_plot"])
    y=utils.read_json(conf["y_plot"])
    plt.figure()
    for key_i in x:
        x_i,y_i=x[key_i],y[key_i]
        plt.text(x_i, 
                 y_i, 
                 key_i,
                 fontdict={'weight': 'bold', 'size': 9})
    x_values=list(x.values())
    y_values=list(y.values())
    plt.title(conf["title"])
    plt.xlabel(conf['x_label'])
    plt.ylabel(conf['y_label'])
    plt.xlim((min(x_values),max(x_values)*1.25))
    plt.ylim((min(y_values),max(y_values)*1.25))
    plt.show()