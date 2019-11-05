from torch import nn


class View(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view((batch_size, *self.output_shape))


class Flatten(View):
    def __init__(self):
        super().__init__((-1,))


class GeneratorCNN(nn.Module):
    def __init__(self, gan_noise_size, gan_output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(gan_noise_size, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128, momentum=0.99, eps=0.001),
            View((2, 8, 8)),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(in_channels=2, out_channels=32, kernel_size=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(32, momentum=0.99, eps=0.001),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(16, momentum=0.99, eps=0.001),
            Flatten(),
            nn.Linear(1024, gan_output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class GeneratorFC(nn.Module):
    def __init__(self, gan_noise_size, gan_output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(gan_noise_size, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128, momentum=0.99, eps=0.001),
            nn.Linear(128, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(32, momentum=0.99, eps=0.001),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(16, momentum=0.99, eps=0.001),
            nn.Linear(1024, gan_output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class DiscriminatorCNN(nn.Module):
    def __init__(self, gan_noise_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(gan_noise_size, 128),
            View((2, 8, 8)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            Flatten(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class DiscriminatorFC(nn.Module):
    def __init__(self, gan_noise_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(gan_noise_size, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(gan_noise_size, 4096),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
