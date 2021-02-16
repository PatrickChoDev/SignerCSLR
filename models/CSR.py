import torch
import torch.nn as nn



class CNN3D(nn.Module):
    def __init__(self, padding_size, num_classes):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 3, 3), padding=(0,1,1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 3, 3), padding=(0,2,2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(1, 2, 2),padding=(0,1,1))
        self.conv5b = nn.Conv3d(512, num_classes, kernel_size=(1, 3, 3))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        return x