import torch
import torch.nn as nn

print("[+] Taking Model Architecture ...")

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, num_anchors=2, split_size=7):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.split_size = split_size
        self.output_size = 7 * 7 * 30

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )

        self.features = nn.Sequential(
            conv_block(3, 192, 7, 2, 3),
            nn.MaxPool2d(2, 2),

            conv_block(192, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(256, 128, 1, 1, 0),
            conv_block(128, 256, 3, 1, 1),
            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(512, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            conv_block(512, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            conv_block(512, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            conv_block(512, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),

            conv_block(512, 512, 1, 1, 0),
            conv_block(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(1024, 512, 1, 1, 0),
            conv_block(512, 1024, 3, 1, 1),
            conv_block(1024, 512, 1, 1, 0),
            conv_block(512, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 2, 1),

            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 1, 1),
        )

        # Final layers
        self.head = nn.Sequential(
            nn.Flatten(),  # [batch, 1024, 7, 7] -> [batch, 50176]
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.output_size)  # Output: [batch, 1470]
        )

    def forward(self, x):
        x = self.features(x)  # -> [B, 1024, 7, 7]
        print("Feature map shape:", x.shape)
        x = self.head(x)      # -> [B, 1470]
        x = x.view(-1, 7, 7, 30)
        return x



# # Example usage
# if __name__ == "__main__":
#     model = YOLOv2(num_classes=20, num_anchors=5)
#     dummy_input = torch.randn(1, 3, 448, 448)  # Standard input size for YOLOv2
#     output = model(dummy_input)
#     print("Output shape:", output.shape)

# from torchinfo import summary

# model = YOLOv1(num_classes=20, num_anchors=2)
# summary(model, input_size=(1, 3, 448, 448))
