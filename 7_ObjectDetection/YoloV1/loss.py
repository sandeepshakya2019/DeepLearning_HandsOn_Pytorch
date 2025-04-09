import torch
import torch.nn as nn
from metrics import intersection_over_union

class YoLoLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoLoLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30] # second bbox
                + (1 - bestbox) * predictions[..., 21:25] # first bob
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

          # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss


# import torch
# from torch import nn

# # Assuming the YoloLoss class is already defined above

# # Hyperparameters
# S = 7   # Grid size
# B = 2   # Number of bounding boxes
# C = 20  # Number of classes
# batch_size = 4

# # Create dummy predictions
# # Shape: (batch_size, S, S, B*5 + C)
# predictions = torch.rand((batch_size, S, S, B * 5 + C))

# # Create dummy targets (same shape)
# # Let's simulate some grid cells having objects (confidence = 1) and some not
# targets = torch.zeros_like(predictions)

# # Randomly set object presence in ~30% of grid cells
# for i in range(batch_size):
#     for row in range(S):
#         for col in range(S):
#             if torch.rand(1).item() < 0.3:
#                 for b in range(B):
#                     targets[i, row, col, b * 5 + 0] = torch.rand(1).item()  # x
#                     targets[i, row, col, b * 5 + 1] = torch.rand(1).item()  # y
#                     targets[i, row, col, b * 5 + 2] = torch.rand(1).item()  # w
#                     targets[i, row, col, b * 5 + 3] = torch.rand(1).item()  # h
#                     targets[i, row, col, b * 5 + 4] = 1                     # confidence
#                 class_label = torch.randint(0, C, (1,))
#                 targets[i, row, col, B * 5 + class_label] = 1

# # Instantiate loss function
# criterion = YoLoLoss(S=S, B=B, C=C)

# # Calculate loss
# loss = criterion(predictions, targets)

# print("Dummy Loss:", loss.item())
