import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
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

def bar_plot(data_dict,
             data,
             clf_types,
             step,
             colors=None):
    fig, ax = plt.subplots()
    color_map=SimpleColorMap(colors)        
    all_values=[]
    for i,data_i in enumerate(data):
        dict_i=data_dict[data_i]
        for j,clf_j in enumerate(clf_types):
            value_j=dict_i[clf_j]
            all_values.append(value_j)
            plt.bar(i*step+j,value_j, 0.4, 
                label = clf_types[j], 
                color= color_map(j))    
    legend_handles = color_map.get_handlers()
    plt.legend(legend_handles,clf_types)
    plt.ylim(*bar_limit(all_values))
    plt.xticks([i*step for i,_ in enumerate(data)], data,rotation='vertical')
    plt.ylabel('Accuracy') 
    plt.show()

def bar_limit(all_values):
    y_min=0.95*np.amin(all_values)
    y_max=1.1*np.amax(all_values)
    if(y_max>1.0):
        return y_min,1.0
    return y_min,y_max

def box_plot(values:list,
             names:list,
             clf_types:list,
             y_label='Accuracy',
             colors=None):
    color_map=SimpleColorMap(colors)        
    unique_clf=list(set(clf_types))
    step=len(unique_clf)
    value_dict={clf_i:[] for clf_i in unique_clf}
    for i,clf_i in enumerate(clf_types):
        value_dict[clf_i].append(values[i])
    fig, ax = plt.subplots()
    for i,clf_i in enumerate(unique_clf):
        positions_i=[j*step+i for j,_ in enumerate(names)]
        box_i=ax.boxplot(value_dict[clf_i],
                         positions=positions_i,
                         patch_artist=True)
        plt.setp(box_i['boxes'], color=color_map(i))
    legend_handles = color_map.get_handlers()
    ax.legend(legend_handles,unique_clf)
    plt.ylabel(y_label)
#    plt.figure(figsize=(7, 3), tight_layout=True)
    offset=int(step/2)
    xticks=[offset + (i*step) for i,_ in enumerate(names)]
    plt.xticks(xticks, names,rotation='vertical')
    plt.tight_layout()
    plt.show()

class SimpleColorMap(object):
    def __init__(self,colors):
        if(colors is None):
            colors=['blue','tomato','lime',
                    'skyblue','peachpuff', 'orange']
        self.colors=colors

    def __call__(self,i):
        return self.colors[i % len(self.colors)]

    def get_handlers(self):
        return [plt.Rectangle((0,0),1,1, color=color_i) 
                    for color_i in self.colors]
    
    def get_color_dict(self,keys):
        return {key_i:self.colors[i] 
                    for i,key_i in enumerate(keys)}

def heatmap(matrix,
            x_labels,
            y_labels,
            title="Statistical significance (RF)"):
    ax=sn.heatmap(matrix,
                  cmap="RdBu_r",
                  linewidth=0.5,
                  cbar=False)
    ax.set_xticklabels(x_labels,rotation = 90)
    ax.set_yticklabels(y_labels,rotation = 0)
    ax.set_title(title)
    plt.show()

def subset_plot(value_dict,data,step=1,colors=None):
    value_dict={key_i:value_dict[key_i] for key_i in data}
    ens_types=[ ens_j
                 for ens_j,_ in list(value_dict.values())[0]]
    data_step={data_i:(i*len(ens_types)*step) 
          for i,data_i in enumerate(value_dict) }
    ens_step={ens_i:(i*step) for i,ens_i in enumerate(ens_types)}
    color_map=SimpleColorMap(colors)
    color_dict=color_map.get_color_dict(ens_types)
    plt.figure()
    min_value,max_value=np.inf,-np.inf
    for data_i,dict_i in value_dict.items():
        for ens_j,value_j in dict_i:
            x_j=data_step[data_i] + ens_step[ens_j]
            min_value= min(min_value,np.min(value_j))
            max_value= max(max_value,np.max(value_j))
            for k,value_k in enumerate(value_j):
                plt.text(x=x_j, 
                         y=value_k, 
                         s=(k+1),
                         color=color_dict[ens_j],
                         fontdict={'weight': 'bold', 'size': 9})
    plt.xlim((0,len(data_step)*len(ens_types)*step+3))
    delta=max_value-min_value
    plt.ylim((min_value,max_value+ delta*0.05))
    labels=data_step.keys()
    xticks=[data_step[key_i] for key_i in data_step]
    plt.xticks(xticks,data,rotation='vertical')
    legend_handles = color_map.get_handlers()
    plt.legend(legend_handles,ens_types)
    plt.title("Clf selection")
    plt.ylabel('Accuracy') 
    plt.show()