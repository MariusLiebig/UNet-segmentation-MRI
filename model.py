import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNETBase(nn.Module):
    def __init__(self, input_channels, output_channels, feature_size, conv, batchnorm, pool, convtranspose):
        super().__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.feature_size = feature_size
        self.conv = conv
        self.batchnorm = batchnorm
        self.pool = pool
        self.convtranspose = convtranspose

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channels = input_channels
        for feature in feature_size:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        for feature in reversed(feature_size):
            self.decoder.append(convtranspose(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))

        self.lowest_layer = self.double_conv(feature_size[-1], feature_size[-1] * 2)
        self.final_conv = self.conv(feature_size[0], output_channels, kernel_size=1)


        # Attention layers
        self.attention_theta = nn.ModuleList()
        self.attention_phi = nn.ModuleList()
        self.attention_psi = nn.ModuleList()
        self.attention_upsample = nn.ModuleList()

        for idx, feature in enumerate(reversed(feature_size)):
            print("Feature size: ", feature)
            print("Idx: ", idx)
            self.attention_theta.append(conv(feature, feature, kernel_size=1, stride=2))
            self.attention_phi.append(conv(feature *2, feature, kernel_size=1, stride=1))
            self.attention_psi.append(conv(feature, 1, kernel_size=1, stride=1))
            self.attention_upsample.append(convtranspose(1, 1, kernel_size=2, stride=2))



    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv(in_channels, out_channels, kernel_size=3, padding=1),
            self.batchnorm(out_channels),
            nn.ReLU(inplace=True),
            self.conv(out_channels, out_channels, kernel_size=3, padding=1),
            self.batchnorm(out_channels),
            nn.ReLU(inplace=True)
        )
    def attention_block(self, skip, g, idx):
        """
        skip : skip connection (from encoder)
        g : gating signal (from decoder)
        idx : which attention block (layer index)
        """

        # Apply predefined conv layers
        theta_skip = self.attention_theta[idx](skip)
        phi_g = self.attention_phi[idx](g)

        if phi_g.shape != theta_skip.shape:
            phi_g = F.interpolate(phi_g, size=theta_skip.shape[2:], mode='trilinear', align_corners=True)
        f = torch.relu(theta_skip + phi_g)
        psi = torch.sigmoid(self.attention_psi[idx](f))

        up = self.attention_upsample[idx](psi)
        out = up * skip
        return out



class UNET(UNETBase):
    def __init__(self, input_channels=1, output_channels=3, feature_size=[32, 64, 128, 256, 512]):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            feature_size=feature_size,
            conv=nn.Conv2d,
            batchnorm=nn.BatchNorm2d,
            pool=nn.MaxPool2d,
            convtranspose=nn.ConvTranspose2d
        )

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(kernel_size=2)(x)
        x = self.lowest_layer(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](x)

        return self.final_conv(x)


class UNET3D(UNETBase):
    def __init__(self, input_channels=1, output_channels=3, feature_size=[64, 128, 256, 512]):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            feature_size=feature_size,
            conv=nn.Conv3d,
            batchnorm=nn.BatchNorm3d,
            pool=nn.MaxPool3d,
            convtranspose=nn.ConvTranspose3d
        )

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(kernel_size=2)(x)
        x = self.lowest_layer(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x_up = self.decoder[i](x) #Upsample
            skip_connection = skip_connections[i // 2]
            if x_up.shape != skip_connection.shape:
                x_up = F.interpolate(x_up, size=skip_connection.shape[2:], mode='trilinear', align_corners=True)
            
            attention = self.attention_block(skip_connection, x, i//2) #x -> ouput before upsampling
            x = torch.cat((attention, x_up), dim=1)
            x = self.decoder[i + 1](x) #Double conv

        return self.final_conv(x)

if __name__ == "__main__":
    model = UNET3D(input_channels=1, output_channels=1, feature_size=[64, 128, 256, 512])
    x = torch.randn((2, 1, 128, 128, 16))  # Example  (B, C, H, W, D)
    y = model(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
