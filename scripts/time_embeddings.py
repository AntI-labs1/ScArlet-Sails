import torch
import torch.nn as nn

class TimeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.hour_emb = nn.Embedding(24, 8)
        self.dow_emb = nn.Embedding(7, 4)
        # Убираем day_of_month - он менее важен
        
    def forward(self, hour, dow, dom):
        h = self.hour_emb(hour)
        d = self.dow_emb(dow)
        return torch.cat([h, d], dim=1)  # 12 dims вместо 18

def add_time_features(df):
    """Добавить временные features в датафрейм"""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day - 1  # ВАЖНО: 0-indexed (1-31 → 0-30)
    return df