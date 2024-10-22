from torch import nn

from guided_diffusion.nn import conv_nd


class DynamicFilterConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_conditions):
        super().__init__()
        # Create filters based on the number of conditions
        self.filters = nn.ModuleList([
            conv_nd(2, in_channels, out_channels, 3, padding=1)
            for _ in range(num_conditions)
        ])

        # A condition selector, which selects filter weights based on the condition
        self.condition_selector = nn.Sequential(
            nn.Linear(in_channels, num_conditions),
            nn.Softmax(dim=-1)  # Ensures the weights sum up to 1
        )

    def forward(self, x, condition):
        # Generate filter weights from the condition
        condition = condition.view(condition.size(0), -1)
        weights = self.condition_selector(condition)

        # Initialize output as zero
        out = 0

        # Apply each filter weighted by its corresponding condition weight
        for i, filt in enumerate(self.filters):
            out += weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1) * filt(x)

        return out
