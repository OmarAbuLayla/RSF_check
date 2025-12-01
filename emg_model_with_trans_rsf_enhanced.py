# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Small helper
# ------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


# ------------------------------------------------------------
# Squeeze-and-Excitation block  (very small, RSF-friendly)
# ------------------------------------------------------------
class SEBlock(nn.Module):
    """# >>> ADDED FOR RSF UPGRADE: lightweight SE block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


# ------------------------------------------------------------
# Positional Encoding (Transformer)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


# ------------------------------------------------------------
# GRU MODULE (enhanced)
# ------------------------------------------------------------
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame

        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers, batch_first=True,
            bidirectional=True
        )

        # >>> ADDED FOR RSF UPGRADE: normalization + dropout
        self.post_gru_norm = nn.LayerNorm(hidden_size * 2)
        self.post_gru_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0),
            self.hidden_size, device=x.device
        )
        out, _ = self.gru(x, h0)

        # >>> ADDED FOR RSF UPGRADE
        out = self.post_gru_norm(out)
        out = self.post_gru_dropout(out)

        if self.every_frame:
            return self.fc(out)
        else:
            return self.fc(out[:, -1, :])


# ------------------------------------------------------------
# MAIN EMG MODEL
# ------------------------------------------------------------
class EMGNet(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=101, frameLen=36, every_frame=True):
        super().__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses

        # ==========================================================
        # >>> ADDED FOR RSF UPGRADE: 1×1 channel-mixing stem
        # ==========================================================
        self.channel_stem = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        stem_out = 32   # becomes new input to fronted2D

        # ==========================================================
        # Convolutional frontend
        # ==========================================================
        self.fronted2D = nn.Sequential(
            nn.Conv2d(stem_out, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # >>> ADDED FOR RSF UPGRADE: SE block
        self.se1 = SEBlock(64)

        self.fronted2D1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fronted2D2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fronted2D3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fronted2D4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.fronted2D5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fronted2D6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.fronted2D7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.5)
        )

        # ==========================================================
        # >>> ADDED FOR RSF UPGRADE: temporal projection block
        # ==========================================================
        self.temporal_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # ==========================================================
        # Backend GRU
        # ==========================================================
        self.gru = GRU(
            input_size=self.inputDim,
            hidden_size=hiddenDim,
            num_layers=2,
            num_classes=nClasses
        )

        # ==========================================================
        # Backend Transformer
        # ==========================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.pe = PositionalEncoding(256)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )
        self.transformer_fc = nn.Linear(256, nClasses)


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------
    def forward(self, x):
        B = x.size(0)

        # >>> ADDED FOR RSF UPGRADE: channel mixing
        x = self.channel_stem(x)

        # conv frontend
        x = self.fronted2D(x)
        x = self.se1(x)      # >>> ADDED FOR RSF UPGRADE
        x = self.fronted2D1(x)
        x = self.fronted2D2(x)
        x = self.fronted2D3(x)
        x = self.fronted2D4(x)
        x = self.fronted2D5(x)
        x = self.fronted2D6(x)
        x = self.fronted2D7(x)

        # final shape = (B, 256, ?, ?)
        x = x.view(B, -1, 256)   # (B, T, 256)

        # >>> ADDED FOR RSF UPGRADE: temporal projection
        x = self.temporal_proj(x)

        if self.mode in ["backendGRU", "finetuneGRU"]:
            return self.gru(x)

        elif self.mode == "transformer":
            x = self.pe(x)
            x = self.transformer(x)
            x = self.transformer_fc(x.mean(dim=1))
            return x

        else:
            raise Exception("Invalid mode selected")


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------
def emg_model(mode, inputDim=256, hiddenDim=512, nClasses=101, frameLen=36, every_frame=True):
    return EMGNet(
        mode, inputDim=inputDim,
        hiddenDim=hiddenDim,
        nClasses=nClasses,
        frameLen=frameLen,
        every_frame=every_frame
    )
