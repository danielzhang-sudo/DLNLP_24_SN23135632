import torch
from B.data import BERTDataset
from torch.utils.data import DataLoader

def test(model, data_test, args):
    device = args.device

    model.eval()
    
    test_accuracy = 0
    test_accuracy_batch = []

    test_data = BERTDataset(data_test)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=True)

    correct_predictions = 0
    num_samples = 0

    with torch.no_grad():
        for batch, data in enumerate(test_loader, 0):
            text, label = data

            ids = text['ids'].squeeze(0).to(device)
            mask = text['mask'].squeeze(0).to(device)
            token_type_ids = text['token_type_ids'].squeeze(0).to(device)

            outputs = model(ids, mask, token_type_ids)

            # validation accuracy
            _, preds = torch.max(outputs, dim=1) # batch dim 
            _, targ = torch.max(label, dim=1)  # batch dim
            num_samples += len(targ)  # technically adding batch size
            correct_predictions += torch.sum(preds == targ)
            correct_predictions_batch = torch.sum(preds == targ)
            test_accuracy_batch.append(correct_predictions_batch/len(targ))
                    
            print(f'Test_Batch {batch}, Test_Acc:  {correct_predictions_batch/len(targ)}')
        
        test_accuracy = float(correct_predictions)/num_samples

    return test_accuracy