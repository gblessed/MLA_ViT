import torch.nn as nn
import torch


class YOLO(nn.Module):
    def __init__(self, img_w = 448, img_h = 448):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h


        ## mybackbone from figure 3 of the paper
        self.backbone = nn.Sequential(

            nn.Conv2d(in_channels = 3, kernel_size=7, out_channels= 192, stride = 2, padding =3),
            


        )


    def forward(self, x):

        return self.backbone(x)



################### testing example######################
# inp = torch.randn((1, 3, 448, 448))
# model = YOLO()
# model(inp).shape
# >> torch.Size([7, 7, 30])