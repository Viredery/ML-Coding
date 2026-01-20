import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            labels: [batch_size]
        Returns:
            torch.Tensor, shape (1,)
        """
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        logits_stabilized = logits - max_logits
        
        # Compute log softmax
        logits_exp = torch.exp(logits_stabilized)
        sum_logits_exp = torch.sum(logits_exp, dim=-1, keepdim=True)
        log_probs = logits_stabilized - torch.log(sum_logits_exp)
        
        labels_one_hot = torch.one_hot(labels, num_classes=logits.shape[-1])

        loss = -torch.sum(log_probs * labels_one_hot, dim=-1)

        return loss.mean()

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [batch_size, feat_dim]
            targets: [batch_size, feat_dim]
        
        Returns:
            torch.Tensor, shape (1,)
        """
        return torch.mean(torch.sum((preds - targets) ** 2, dim=-1))


class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [batch_size, feat_dim] (should be probabilities)
            targets: [batch_size, feat_dim] (should be probabilities)
        
        Returns:
            torch.Tensor, shape (1,)
        """
        eps = 1e-8
        
        log_preds = torch.log(preds + eps)
        log_targets = torch.log(targets + eps)
        
        return torch.mean(torch.sum(targets * (log_targets - log_preds), dim=-1))


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: [batch_size, num_classes] (Expects Log-Probabilities, e.g. from LogSoftmax)
            labels: [batch_size, num_classes] (One-hot)
        
        Returns:
            torch.Tensor, shape (1,)
        """
        return -torch.mean(torch.sum(labels * log_probs, dim=-1))
