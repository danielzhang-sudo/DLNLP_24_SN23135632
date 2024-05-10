import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

def to_latex(true_labels, pred):
    #print(true_labels)
    #print(pred)

    report = classification_report(true_labels, pred, output_dict=True)
    
    df = pd.DataFrame.from_dict(report).T
    with open('mytable.tex', 'w') as f:
        f.write(df.to_latex())

def plot_figures(data1, data2, filename):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(data1, label='train')
    plt.plot(data2, label='validation')
    plt.title(filename)
    plt.legend()
    fig.savefig(filename+'.png', dpi=fig.dpi)

def plot_cm(y_true, y_pred):
    #print(y_true)
    #print(y_pred)
    #print(type(y_pred))
    #print(y_pred.size)
    labels = ['ambience','anecdotes/miscellaneous','food','price','service']
    conf_mat_dict={}

    for label_col in range(len(labels)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        print(y_true_label)
        print(y_pred_label)

        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

    num_labels = len(conf_mat_dict)
    fig, axes = plt.subplots(1, num_labels, figsize=(25, 5))
   
    for i, (label, conf_matrix) in enumerate(conf_mat_dict.items()):
        ax = axes[i]
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax, fmt='g')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix for Label: {label}')

    plt.tight_layout()
    plt.savefig('conf_matrices.png')

    #cm = multilabel_confusion_matrix(true_labels, pred)
    #plt.figure(figsize=(8, 6))
    #plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    #plt.title('Confusion Matrix')
    #plt.colorbar()
    #plt.savefig('confusion_matrix.png')

def plot_roc(true_labels, pred):
    fpr, tpr, thresholds = roc_curve(true_labels, pred)
    roc_auc = auc(fpr, tpr)


    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
