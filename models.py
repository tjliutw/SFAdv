import torch
import torchvision

class Encoder(torch.nn.Module):
    # input shape (B, c_in, H, W)
    # output shape (B, c_out)
    def __init__(self, c_in=1, nf=32, c_out=10, H=32, W=16):
        super(Encoder, self).__init__()
        
        BIAS = True
        self.main = torch.nn.Sequential(
            # (B, c_in, H, W)
            torch.nn.Conv2d(in_channels=c_in, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=BIAS),
            torch.nn.BatchNorm2d(num_features=nf),
            torch.nn.LeakyReLU(0.2),
            # (B, nf, H//2, W//2)
            torch.nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=4, stride=2, padding=1, bias=BIAS),
            torch.nn.BatchNorm2d(num_features=nf*2),
            torch.nn.LeakyReLU(0.2),
            # (B, nf*2, H//4, W//4)
            torch.nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=4, stride=2, padding=1, bias=BIAS),
            torch.nn.BatchNorm2d(num_features=nf*4),
            torch.nn.LeakyReLU(0.2)
            # (B, nf*4, H//8, W//8)
        )
        
        self.linear = torch.nn.Sequential(
            # (B, nf*4 * H//8 * W//8)
            torch.nn.Linear(in_features=nf*4*H//8*W//8, out_features=512, bias=BIAS),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.5),
            # (B, 512)
            torch.nn.Linear(in_features=512, out_features=256, bias=BIAS),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.5),
            # (B, 256)
            torch.nn.Linear(in_features=256, out_features=128, bias=BIAS),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.5),
            # (B, 128)
            torch.nn.Linear(in_features=128, out_features=c_out, bias=BIAS)
            # (B, c_out)
        )
        
        self.initialize_weights()
    # end of __init__()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
    # end of initialize_weights()

    def forward(self, x):
        z = self.main(x)
        z = z.view(len(z), -1)
        output = self.linear(z)
        return output
    # end of forward()
# end of class Encoder

class Decoder(torch.nn.Module):
    def __init__(self, input_size=10, nf=32, content_size=10, H=32, W=16):
        super(Decoder, self).__init__()
        self.content_size = content_size
        self.H = H
        self.W = W

        BIAS = True
        self.linear1 = torch.nn.Sequential(
            # (B, input_size)
            torch.nn.Linear(input_size, 128, bias=BIAS),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            # (B, 128)
        )
        
        self.linear2 = torch.nn.Sequential(
            # (B, 128+content_size)
            torch.nn.Linear(128+content_size, 256, bias=BIAS),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            # (B, 256)
        )
        
        self.linear3 = torch.nn.Sequential(
            # (B, 256+content_size)
            torch.nn.Linear(256+content_size, 512, bias=BIAS),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            # (B, 512)
        )
        
        self.linear4 = torch.nn.Sequential(
            # (B, 512+content_size)
            torch.nn.Linear(512+content_size, nf*4*H//8*W//8, bias=BIAS),
            torch.nn.BatchNorm1d(nf*4*H//8*W//8),
            torch.nn.ReLU()
            # (B, nf*4 * H//8 * W//8)
        )
        
        self.layers1 = torch.nn.Sequential(
            # (B, nf*4+content_size, H//8, W//8)
            torch.nn.ConvTranspose2d(nf*4+content_size, nf*2, kernel_size=4, stride=2, padding=1, bias=BIAS),
            torch.nn.BatchNorm2d(nf*2),
            torch.nn.ReLU(),
            # (B, nf*2, H//4, W//4)
        )
        
        self.layers2 = torch.nn.Sequential(
            # (B, nf*2+content_size, H//4, W//4)
            torch.nn.ConvTranspose2d(nf*2+content_size, nf, kernel_size=4, stride=2, padding=1, bias=BIAS),
            torch.nn.BatchNorm2d(nf),
            torch.nn.ReLU(),
            # (B, nf, H//2, W//2)
        )
        
        self.layers3 = torch.nn.Sequential(
            # (B, nf+content_size, H//2, W//2)
            torch.nn.ConvTranspose2d(nf+content_size, nf//2, kernel_size=4, stride=2, padding=1, bias=BIAS),
            torch.nn.BatchNorm2d(nf//2),
            torch.nn.ReLU(),
            # (B, nf//2, H, W)
            torch.nn.ConvTranspose2d(nf//2, 1, kernel_size=3, stride=1, padding=1)
            # (B, out_dims, H, W)
        )
        
        self.initialize_weights()
    # end of __init__()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)  
    # edn of initialize_weights()

    def forward(self, style, content):
        z_y = torch.cat((style, content), 1)
        z = self.linear1(z_y)
        z_y = torch.cat((z, content), 1)
        z = self.linear2(z_y)
        z_y = torch.cat((z, content), 1)
        z = self.linear3(z_y)
        z_y = torch.cat((z, content), 1)
        z = self.linear4(z_y)       # (B, nf*4 * H//8 * W//8)
        
        z = z.view(len(z), -1, self.H//8, self.W//8)    # (B, nf*4, H//8, W//8)
        y = content.unsqueeze(2).unsqueeze(3)           # (B, content_size, 1, 1)
        y_feature_map = y.expand(y.size(0), self.content_size, z.size(2), z.size(3))  # (B, content_size, H//8, W//8)
        z_y = torch.cat((z, y_feature_map), 1)          # (B, nf*4+content_size, H//8, W//8)
        z = self.layers1(z_y)                           # (B, nf*2, H//4, W//4)
        
        y_feature_map = y.expand(y.size(0), self.content_size, z.size(2), z.size(3))   # (B, content_size, H//4, W//4)
        z_y = torch.cat((z, y_feature_map), 1)          # (B, nf*2+content_size, H//4, W//4)
        z = self.layers2(z_y)                           # (B, nf, H//2, W//2)
        
        y_feature_map = y.expand(y.size(0), self.content_size, z.size(2), z.size(3))   # (B, content_size, H//2, W//2)
        z_y = torch.cat((z, y_feature_map), 1)          # (B, nf+content_size, H//2, W//2)
        z = self.layers3(z_y)           # out image     # (B, out_dims, H, W)

        return z
    # end of forward()
# end of class Decoder

class Generator(torch.nn.Module):
    def __init__(self, content_size=10, style_size=10, c_in=1, nf=32, H=32, W=16, add_mask=True, add_noise=True, noise_weight=0.1):  # noise size = content_size
        super(Generator, self).__init__()
        self.H = H
        self.W = W
        self.content_size = content_size
        self.add_mask = add_mask
        self.maskMin = -0.9450980424880981
        self.mask_threshold = -0.945098
        self.add_noise = add_noise
        self.noise_weight = noise_weight
        
        self.ce = Encoder(c_in=c_in, nf=nf, c_out=content_size, H=H, W=W)
        self.se = Encoder(c_in=c_in, nf=nf, c_out=style_size, H=H, W=W)
        self.decoder = Decoder(input_size=content_size+style_size, nf=nf, content_size=content_size, H=H, W=W)
    # end of __init__()
    
    def do_add_mask(self, mask_sources, inputs, progress_steps=None):
        if self.add_mask:
            mask1 = mask_sources == -1
            mask2 = mask_sources <= self.mask_threshold
            mask2 = mask2 & ~mask1
            if progress_steps is not None:
                mask1 = mask1.repeat_interleave(progress_steps, dim=0).view(len(mask_sources)*progress_steps, 1, self.H, self.W)
                mask2 = mask2.repeat_interleave(progress_steps, dim=0).view(len(mask_sources)*progress_steps, 1, self.H, self.W)
            inputs[mask1] = -1
            inputs[mask2] = self.maskMin
            return inputs
        else:
            return inputs
    # end of do_add_mask()
    
    def do_add_noise(self, inputs):
        if self.add_noise:
            noise = torch.randn(inputs.shape[0], inputs.shape[1], device=inputs.device) * self.noise_weight
            inputs = torch.add(inputs, noise)
            return inputs
        else:
            return inputs
    # end of do_add_noise()
    
    def forward(self, x_inputs):
        x_content = self.ce(x_inputs)
        x_content = self.do_add_noise(x_content)
        
        x_style = self.se(x_inputs)
        x_recons = self.decoder(x_style, x_content)
        x_recons = self.do_add_mask(x_inputs, x_recons)
        
        return x_content, x_style, x_recons
    # end of forward()
    
    def get_adv(self, x_inputs, t_inputs, fusion_ratio=1.0):
        x_content = self.ce(x_inputs)
        x_content = self.do_add_noise(x_content)
        
        # reconstruct
        x_style = self.se(x_inputs)
        x_recons = self.decoder(x_style, x_content)
        x_recons = self.do_add_mask(x_inputs, x_recons)
        
        # change style
        t_style = self.se(t_inputs)
        fusion_style = x_style * (1.0 - fusion_ratio) + t_style * fusion_ratio
        x_adv = self.decoder(fusion_style, x_content)
        x_adv = self.do_add_mask(x_inputs, x_adv)
        
        return x_recons, x_adv
    # end of get_adv()
# end of class Generator

class Discriminator(torch.nn.Module):
    def __init__(self, c_in=1, nf=32, H=32, W=16, recon_level=2):
        super(Discriminator, self).__init__()
        self.recon_level = recon_level
        self.conv = torch.nn.ModuleList([
            # (B, c_in, H, W)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=c_in, out_channels=nf, kernel_size=5, stride=1, padding=2, bias=False),
                torch.nn.LeakyReLU(0.2)
                # (B, nf, H, W)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=nf*2),
                torch.nn.LeakyReLU(0.2)
                # (B, nf*2, H//2, W//2)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=nf*4),
                torch.nn.LeakyReLU(0.2)
                # (B, nf*4, H//4, W//4)
            )
        ])
        self.fc = torch.nn.Sequential(
            # (B, nf*4*H//4*W//4)
            torch.nn.Linear(in_features=nf*4*H//4*W//4, out_features=128, bias=False),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=0.5),
            # (B, 128)
            torch.nn.Linear(in_features=128, out_features=1)
            # (B, 1)
        )
        
        self.initialize_weights()
    # end of __init__()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    # end of initialize_weights()

    def forward(self, x_recon, x_original, mode='RECON'):
        z = torch.cat((x_recon, x_original), 0)
        if mode == "RECON":
            for i, lay in enumerate(self.conv):
                if i == self.recon_level:
                    z = lay(z)
                    layer_repre = z.view(len(z), -1)
                    return layer_repre
                else:
                    z = lay(z)
        else:
            for lay in self.conv:
                z = lay(z)
            z = z.view(len(z), -1)
            output = self.fc(z)
            return torch.sigmoid(output)
    # end of forward()
# end of class Discriminator

class LeNet5(torch.nn.Module):
    # network structure
    def __init__(self, c_in=1, c_out=10, H=32, W=16):
        super(LeNet5, self).__init__()
        self.layers1 = torch.nn.Sequential(
            # (B, c_in, H, W)
            torch.nn.Conv2d(in_channels=c_in, out_channels=6, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            # (B, 6, H, W)
            torch.nn.MaxPool2d(kernel_size=2),
            # (B, 6, H//2, W//2)
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            torch.nn.ReLU(),
            # (B, 16, (H//2-4), (W//2-4))  # 卷積後 H 和 W 會減少 4
            torch.nn.MaxPool2d(kernel_size=2),
            # (B, 16, (H//4-2), (W//4-2))  # 池化後再減半
        )
        
        self.liner1 = torch.nn.Sequential(
            # (B, 16, (H//4-2), (W//4-2))
            torch.nn.Linear(in_features=16*(H//4-2)*(W//4-2), out_features=120),
            torch.nn.ReLU(),
            # (B, 120)
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            # (B, 84)
            torch.nn.Linear(in_features=84, out_features=c_out),
            # (B, c_out)
        )
        
        self.initialize_weights()
    # end of __init__()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    # end of initialize_weights()

    def forward(self, x):
        x = self.layers1(x)
        x = x.view(x.shape[0], -1)
        x = self.liner1(x)
        return x
    # end of forward()
# end of class LeNet5

class VGG11(torch.nn.Module):   # This class only for (1, 32, 32) and input (1, 32, 16)
    def __init__(self, c_in=1, c_out=10, H=32, W=32):   
        super(VGG11, self).__init__()        
        self.features = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=1),  
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),       # 32x16 → 16x8
            # (B, 64, H//2, W//2)

            # Block 2
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),       # 16x8 → 8x4
            # (B, 128, H//4, W//4)

            # Block 3
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),       # 8x4 → 4x2
            # (B, 256, H//8, W//8)

            # Block 4
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),       # 4x2 → 2x1
            # (B, 512, H//16, W//16)

            # Block 5 (optional: included in full VGG but might not help much here) 
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),       
            # (B, 512, H//32, W//32)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512 * (H//32) * (W//32), out_features=4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),

            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),

            torch.nn.Linear(in_features=4096, out_features=c_out)
        )
        
        self.initialize_weights()
    # end of __init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    # end of initialize_weights()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.features(x)  
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    # end of forward()
# end of class VGG11

class SmallAlexNet(torch.nn.Module):
    def __init__(self, c_in=1, c_out=10, H=32, W=16):
        super(SmallAlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (B, 1, H, W) -> (B, 64, H, W)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),                 # (B, 64, H, W) -> (B, 64, H//2, W//2)
            
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),          # (B, 64, H//2, W//2) -> (B, 192, H//2, W//2)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),                 # (B, 192, H//2, W//2) -> (B, 192, H//4, W//4)
            
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),         # (B, 192, H//4, W//4) -> (B, 384, H//4, W//4)
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),         # (B, 384, H//4, W//4) -> (B, 256, H//4, W//4)
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),         # (B, 256, H//4, W//4) -> (B, 256, H//4, W//4)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),                 # (B, 256, H//4, W//4) -> (B, 256, H//8, W//8)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * (H//8) * (W//8), 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, c_out),
        )
        
        self.initialize_weights()
    # end of __init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    # end of initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    # end of forward()
# end of class SmallAlexNet

class ResNet18(torch.nn.Module):    # This class only for (1, 32, 32) and input (1, 32, 16)
    def __init__(self, c_in=1, c_out=10, H=32, W=32):   
        super(ResNet18, self).__init__()
        self.c_in = c_in     
        self.net = torchvision.models.resnet18(weights=None, num_classes=c_out)
        self.net.conv1 = torch.nn.Conv2d(c_in, 64, kernel_size=7, stride=2, padding=3, bias=False)      
        self.initialize_weights()
    # end of __init__()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    # end of initialize_weights()
    
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        pred = self.net(x)
        return pred
    # end of forward()
# end of class ResNet

