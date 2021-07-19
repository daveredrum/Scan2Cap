import torch
import torch.nn as nn
from torchvision import models


class ResNet101NoFC(nn.Module):
    def __init__(self, pretrained, progress, device, mode):
        super(ResNet101NoFC, self).__init__()
        image_modules = list(models.resnet101(pretrained=pretrained, progress=progress).children())[
                        :-1]  # all layer expect last layer
        self.resnet = nn.Sequential(*image_modules)
        self.device = device
        self.mode = mode

    def forward(self, image, boxes, object_ids):
        image = torch.stack(image).to(self.device)
        if self.mode == 'bbox2feat':
            return self.forward_image_boxes(image=image, boxes=boxes, object_ids=object_ids)
        if self.mode == 'frame2feat':
            return self.forward_image(image=image)

    def forward_image_boxes(self, image, boxes, object_ids):
        batch_size = len(image)
        batch_feats = []
        for i in range(batch_size):
            num_boxes = len(boxes[i])
            frame_object_features = {}
            for j in range(num_boxes):
                x_min = int(boxes[i][j][0].item())
                y_min = int(boxes[i][j][1].item())
                x_max = int(boxes[i][j][2].item())
                y_max = int(boxes[i][j][3].item())
                cropped_tensor = image[i][:, y_min:y_max, x_min:x_max]
                cropped_tensor = cropped_tensor.unsqueeze(0)
                try:
                    f = self.resnet(cropped_tensor.to(self.device))
                except BaseException as be:
                    print(be)
                    print("target box: ",boxes[i][j])
                    print("image: ", image[i].shape)
                    print("cropped: ", cropped_tensor.shape)
                bbox_object_id = int(object_ids[i][j].item())
                frame_object_features[str(bbox_object_id)] = f

            batch_feats.append(frame_object_features)

        return batch_feats

    def forward_image(self, image):
        return self.resnet(image)
