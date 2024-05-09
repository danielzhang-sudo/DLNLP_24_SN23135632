from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

def ACD_loss(true, pred):
    loss = BCEWithLogitsLoss()
    return loss(true, pred)

def ACSA_loss(true, pred):
    """
    Calculates the combined categorical cross-entropy loss for a multiclass multilabel problem.
    
    Args:
        output (torch.Tensor): The output tensor of shape (n, 5).
        target (torch.Tensor): The target tensor of shape (n, 5) with integer class indices.
    
    Returns:
        torch.Tensor: The combined loss.
    """
    batch_size, num_labels = pred.shape
    
    # Calculate the categorical cross-entropy loss for each element in the output
    ce_losses = []
    for i in range(num_labels):
        ce_loss = CrossEntropyLoss()(true[:, i], pred[:, i])
        ce_losses.append(ce_loss)
    
    # Take the average of the losses
    combined_loss = sum(ce_losses) / num_labels
    
    return combined_loss