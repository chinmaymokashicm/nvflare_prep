import torch
import torch.nn as nn
from torchsummary import summary
import time, os


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False):
        super(Conv3DBlock, self).__init__()
        # If current block is bottleneck, do not apply pooling
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not self.bottleneck:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            
    def forward(self, x):
        x_prepool = self.conv1(x)
        x_prepool = self.bn1(x_prepool)
        x_prepool = self.relu(x_prepool)
        
        x_prepool = self.conv2(x_prepool)
        x_prepool = self.bn2(x_prepool)
        x_prepool = self.relu(x_prepool)
        
        if not self.bottleneck:
            x = self.pool(x_prepool)
        else:
            x = x_prepool
        
        return x, x_prepool
    

class UpConv3dBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None):
        super(UpConv3dBlock, self).__init__()
        # If the block is the last layer, add final convolution and return prediction
        # res_channels -> Number of channels from the skip connections
        
        if last_layer and num_classes is None:
            raise Exception("Invalid arguments!")
        
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=res_channels+in_channels, out_channels=in_channels//2, kernel_size=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=2, padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=1)
            
    def forward(self, x, skip=None):
        # skip -> skip connection
        x = self.upconv1(x)
        if skip is not None:
            # Adjust the spatial dimensions of x to match the size of skip
            x = nn.functional.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=False)
            
            # print("x:", x.size())
            # print("skip:", skip.size())
            x = torch.cat((x, skip), dim=1)
        
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.last_layer:
            x = self.conv3(x)
            
        return x
    
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512):
        super(UNet3D, self).__init__()
        
        self.e1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.e2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.e3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        
        self.bottleneck = Conv3DBlock(in_channels=level_channels[2], out_channels=bottleneck_channel, bottleneck=True)
        
        self.d1 = UpConv3dBlock(in_channels=bottleneck_channel, res_channels=level_channels[2])
        self.d2 = UpConv3dBlock(in_channels=level_channels[2], res_channels=level_channels[1])
        self.d3 = UpConv3dBlock(in_channels=level_channels[1], res_channels=level_channels[0], num_classes=num_classes, last_layer=True)
        
    def forward(self, x):
        # Encoder
        # print("1:", x.size(), skip1.size())
        x, skip1 = self.e1(x)
        # print("1:", x.size(), skip1.size())
        x, skip2 = self.e2(x)
        # print("2:", x.size(), skip2.size())
        x, skip3 = self.e3(x)
        # print("3:", x.size(), skip3.size())
        x, _ = self.bottleneck(x)
        # print("4:", x.size())
        
        # Decoder
        # print("x:", x.size(), "skip3:", skip3.size())
        x = self.d1(x, skip3)
        # print("x:", x.size(), "skip2:", skip2.size())
        x = self.d2(x, skip2)
        # print("x:", x.size(), "skip1:", skip1.size())
        x = self.d3(x, skip1)
        # print("x:", x.size())
        
        return x

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    model = UNet3D(in_channels=1, num_classes=1)
    start_time = time.time()
    # summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    x = torch.rand((1, 1, 572, 572, 128))
    y = model.forward(x)
    print(y)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(y.size())
    
    """
    CPU - 187.04266715049744 seconds
    GPU (0,1,2) - 260.1612582206726
    """