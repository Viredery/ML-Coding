import torch

class ClipLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # create a learnable parameter t
        self.t = torch.nn.Parameter(torch.tensor(0.0))

        self.image_cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.text_cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the clip loss.

        Args:
            image_embeds: torch.Tensor, shape (batch_size, num_ins, embedding_dim)
            text_embeds: torch.Tensor, shape (batch_size, num_ins, embedding_dim)

        Returns:
            torch.Tensor, shape (1,)
        """

        batch_size, num_instances, _ = image_embeds.shape

        image_embeds = image_embeds.flatten(0, 1)
        text_embeds = text_embeds.flatten(0, 1)

        normed_image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        normed_text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)

        similarity = normed_image_embeds @ normed_text_embeds.transpose(0, 1)
        similarity = similarity * torch.exp(self.t)
        labels = torch.arange(batch_size * num_instances, dtype=torch.long, device=image_embeds.device)
        
        image_loss = self.image_cross_entropy_loss(similarity, labels)
        text_loss = self.text_cross_entropy_loss(similarity.transpose(0, 1), labels)

        return (image_loss + text_loss) / 2.0


if __name__ == '__main__':
    image_embeds = torch.randn(2, 3, 512)
    text_embeds = torch.randn(2, 3, 512)
    loss = ClipLoss()
    print(loss(image_embeds, text_embeds))
