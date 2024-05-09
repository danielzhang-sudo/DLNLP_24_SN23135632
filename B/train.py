import torch
from B.data import BERTDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from B.utils import plot_figures
from B.config import tokenizer
from torch.utils.data import DataLoader

def train(model, data_train, data_val, loss_fn, args):

    print(['#']*15)
    print('begin training...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs

    train_accuracy = []
    train_accuracy_batch = []
    train_losses = []
    train_losses_batch = []

    val_accuracy = []
    val_accuracy_batch = []
    val_losses = []
    val_losses_batch = []

    training_data = BERTDataset(X=data_train.iloc[:, :1], y=data_train.iloc[:, 4:], tokenizer=tokenizer)
    val_data = BERTDataset(X=data_val.iloc[:, :1], y=data_val.iloc[:, 4:], tokenizer=tokenizer)

    training_loader = DataLoader(training_data, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_data, batch_size= 5, shuffle=True)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    steps = len(training_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    
    for epoch in range(epochs):
        model.train()

        batch_train_loss = 0

        correct_train_predictions = 0
        num_train_samples = 0

        for _, data in enumerate(training_loader, 0):

            ids, mask, token_type_ids, label = data

            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)

            outputs = model(ids.to(device), mask.to(device), token_type_ids.to(device))

            optimizer.zero_grad()
            loss = loss_fn(outputs, label)
            batch_train_loss += loss
            train_losses_batch.append(loss/len(training_loader))

            # training accuracy
            _, preds = torch.max(outputs, dim=1) # batch dim 
            _, targ = torch.max(label, dim=1)  # batch dim
            num_train_samples += len(targ)  # technically adding batch size
            correct_train_predictions += torch.sum(preds == targ)
            correct_predictions_batch = torch.sum(preds == targ)
            train_accuracy_batch.append(correct_predictions_batch/len(targ))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_losses.append(batch_train_loss/len(training_loader))
        train_accuracy.append(float(correct_train_predictions)/num_train_samples)
        print(f'Epoch: {epoch}, Train_Loss:  {loss.item()}, Train_Acc:  {float(correct_train_predictions)/num_train_samples}')

        batch_val_loss = 0

        correct_predictions = 0
        num_val_samples = 0
        
        model.eval()
        with torch.no_grad():
            for batch, data in enumerate(val_loader, 0):

                ids, mask, token_type_ids, label = data

                ids = ids.to(device)
                mask = mask.to(device)
                token_type_ids = token_type_ids.to(device)
                label = label.to(device)

                outputs = model(ids, mask, token_type_ids)
                
                loss = loss_fn(outputs, label)
                batch_val_loss += loss
                val_losses_batch.append(loss/len(val_loader))

                # validation accuracy
                _, preds = torch.max(outputs, dim=1) # batch dim 
                _, targ = torch.max(label, dim=1)  # batch dim
                num_val_samples += len(targ)  # technically adding batch size
                correct_predictions += torch.sum(preds == targ)
                correct_predictions_batch = torch.sum(preds == targ)
                val_accuracy_batch.append(correct_predictions_batch/len(targ))
                        
            
            val_losses.append(batch_val_loss/len(val_loader))
            val_accuracy.append(float(correct_predictions)/num_val_samples)
            print(f'Epoch: {epoch}, Val_Loss:  {loss.item()}, Val_Acc:  {float(correct_predictions)/num_val_samples}')
        
        if val_accuracy[epoch] >= train_accuracy[epoch]:
            torch.save(model.state_dict(), f'best_epoch_{epoch}_weights.pth')
        else:
            torch.save(model.state_dict(), f'{epoch}_epoch_weights.pth')

    plot_figures(train_losses.cpu().numpy(), 'train_losses.png')
    plot_figures(train_accuracy.cpu().numpy(), 'train_accuracy.png')
    plot_figures(val_losses.cpu().numpy(), 'val_losses.png')
    plot_figures(val_accuracy.cpu().numpy(), 'val_accuracy.png')
    
    return train_accuracy[-1]

