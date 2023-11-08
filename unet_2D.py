import torch
import torch.nn as nn
    
def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

# Convolution block used in encoder and decoder
class UNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        
        # Create the skeleton
        
        # Encoder/downsampling: captures the context and high-level features
        self.encoder_conv1 = UNetConv2DBlock(in_channels, 64)
        self.encoder_conv2 = UNetConv2DBlock(64, 128)
        self.encoder_conv3 = UNetConv2DBlock(128, 256)
        self.encoder_conv4 = UNetConv2DBlock(256, 512)
        self.encoder_conv5 = UNetConv2DBlock(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder/upsampling: reconstructs the segmentation mask from the encoded features
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv1 = UNetConv2DBlock(1024, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv2 = UNetConv2DBlock(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv3 = UNetConv2DBlock(256, 128)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv4 = UNetConv2DBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1_prepool = self.encoder_conv1(x)
        e1 = self.pool(e1_prepool)

        e2_prepool = self.encoder_conv2(e1)
        e2 = self.pool(e2_prepool)

        e3_prepool = self.encoder_conv3(e2)
        e3 = self.pool(e3_prepool)

        e4_prepool = self.encoder_conv4(e3)
        e4 = self.pool(e4_prepool)
        
        e0 = self.encoder_conv5(e4)
        
        # Decoder
        d1 = self.upconv1(e0)
        d1 = torch.cat((crop_tensor(e4_prepool, d1), d1), dim=1)
        d1 = self.decoder_conv1(d1)
        
        d2 = self.upconv2(d1)
        d2 = torch.cat((crop_tensor(e3_prepool, d2), d2), dim=1)
        d2 = self.decoder_conv2(d2)
        
        d3 = self.upconv3(d2)
        d3 = torch.cat((crop_tensor(e2_prepool, d3), d3), dim=1)
        d3 = self.decoder_conv3(d3)
        
        d4 = self.upconv4(d3)
        d4 = torch.cat((crop_tensor(e1_prepool, d4), d4), dim=1)
        d4 = self.decoder_conv4(d4)
        
        return self.out_conv(d4)
        
        
if __name__ == "__main__":
    x = torch.rand((1, 3, 572, 572))
    model = UNet2D(in_channels=3, out_channels=1)
    model.forward(x)