import pandas as pd
import matplotlib.pyplot as plt

def to_latex(report):
    df = pd.DataFrame.fromm_dict(report)
    with open('mytable.tex', 'w') as f:
        f.write(df.to_latex())

def plot_figures(data1, data2, filename):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(data1, label='train')
    plt.plot(data2, label='validation')
    plt.title(filename)
    plt.legend()
    fig.savefig(filename+'.png', dpi=fig.dpi)
