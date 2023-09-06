from model.mda_parts import *

class MDANet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MDANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvWithAtt(n_channels, 64)
        self.down1 = DownWithAtt(64, 128)
        self.down2 = DownWithAtt(128, 256)
        self.down3 = DownWithAtt(256, 512)
        self.down4 = DownWithAtt(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # CONV+BN+RELU+(optional)Upsample+MDA
        self.side1 = Conv_Up(64, 16, 0) # No need to upsample
        self.side2 = Conv_Up(64, 16, 2) #
        self.side3 = Conv_Up(128,16, 4)
        self.side4 = Conv_Up(256,16, 8)

        self.score1 = OutConv(16, n_classes) # side1
        self.score2 = OutConv(16, n_classes) # side2
        self.score3 = OutConv(16, n_classes) # side3
        self.score4 = OutConv(16, n_classes) # side4
    
        self.score5 = OutConv(16, n_classes)  # fuse(side1, ..., side4)

    def forward(self, x):
        x1 = self.inc(x)    # 64 H W

        x2 = self.down1(x1) # 128 H/2 W/2
        x3 = self.down2(x2) # 256 H/4 W/4
        x4 = self.down3(x3) # 512 H/8 W/8
        x5 = self.down4(x4) # 512 H/16 W/16

        x6 = self.up1(x5, x4) # 256 H/8 W/8
        x7 = self.up2(x6, x3) # 128 H/4 W/4
        x8 = self.up3(x7, x2) # 64 H/2 W/2
        x9 = self.up4(x8, x1) # 64 H W

        side4 = self.side4(x6) # 16 H W
        side3 = self.side3(x7) # 16 H W
        side2 = self.side2(x8) # 16 H W
        side1 = self.side1(x9) # 16 H W

        score1 = self.score1(side1)
        score2 = self.score2(side2)
        score3 = self.score3(side3)
        score4 = self.score4(side4)

        #score5 = self.score5(torch.cat([score1, score2, score3, score4], dim=1))
        side5 = side1 + side2 + side3 +side4 
        score5 = self.score5(side5)

        return score1, score2, score3, score4, score5