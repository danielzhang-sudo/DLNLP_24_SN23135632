import torch
import numpy as np
from B.data import BERTDataset
from B.config import tokenizer
from B.utils import plot_cm, plot_roc, to_latex
from torch.utils.data import DataLoader

def test(model, data_test, task, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    
    test_accuracy = 0
    test_accuracy_batch = []
    predictions = []
    true_labels = []

    test_data = BERTDataset(X=data_test.iloc[:, :1], y=data_test.iloc[:, 4:], tokenizer=tokenizer)
    print(test_data[0])
    test_loader = DataLoader(test_data, batch_size=5, shuffle=True)

    correct_predictions = 0
    num_samples = 0

    with torch.no_grad():
        for batch, data in enumerate(test_loader, 0):
            ids, mask, token_type_ids, label =  data

            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)

            outputs = model(ids, mask, token_type_ids)

            # validation accuracy
            _, preds = torch.max(outputs, dim=1) # batch dim 
            _, targ = torch.max(label, dim=1)  # batch dim
            
            print(preds[0])
            print(preds[0])
            # predictions.append(preds.cpu().detach().to(torch.int).numpy())
            # true_labels.append(targ.cpu().detach().to(torch.int).numpy())
            num_samples += len(targ)  # technically adding batch size
            correct_predictions += torch.sum(preds == targ)
            correct_predictions_batch = torch.sum(preds == targ)
            test_accuracy_batch.append(correct_predictions_batch/len(targ))

            #_, pred = torch.max(outputs, 1)
            predictions.extend(outputs.cpu().detach().numpy())
            true_labels.extend(label.cpu().detach().numpy())

        # plot_cm(np.array(true_labels), np.array(predictions))
        # plot_roc(np.array(true_labels), np.array(predictions))
        # to_latex(np.array(true_labels), np.array(predictions))

        # print(f'Test_Acc:  {correct_predictions_batch/len(targ)}')
        if task == 'a':
            preds = (torch.tensor(predictions) > 0.5).int()
            plot_cm(np.array(true_labels), np.array(preds))
            # plot_roc(np.array(true_labels), np.array(preds))
            
            to_latex(np.array(true_labels), np.array(preds))
        test_accuracy = float(correct_predictions)/num_samples

    return test_accuracy
