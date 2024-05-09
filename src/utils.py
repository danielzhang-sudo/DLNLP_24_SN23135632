import pandas as pd
import matplotlib.pyplot as plt

def to_latex(report):
    df = pd.DataFrame.fromm_dict(report)
    with open('mytable.tex', 'w') as f:
        f.write(df.to_latex())

def plot_figures(x_data, filename):

    fig = plt.figure(figsize=(10, 7))
    plt.plot(x_data)
    fig.savefig(filename, dpi=fig.dpi)
