# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
# 确保数据按时间排序
df = df.sort_values('datetime').reset_index(drop=True)

# 生成安全的时间范围
start_time = df['datetime'].min()
end_time = df['datetime'].max()

# 生成每日时间序列（使用安全的时间范围）
time_index = pd.date_range(
    start=start_time,
    end=end_time,
    freq='24h'  # 24小时频率，减少运算量，更具统计意义。
)

# 根据生成的时间序列选取数据（前向填充处理缺失）
df_downsampled = df.set_index('datetime').reindex(time_index, method='ffill')

# 过滤无效数据（时间索引超出原始数据范围时会出现NaN）
df_downsampled = df_downsampled.dropna(subset=df.drop(columns=['datetime']).columns)

# 恢复原始列结构（关键修复）
df_downsampled.reset_index(inplace=True)
df_downsampled.columns = ['datetime'] + df.drop(columns=['datetime']).columns.tolist()

# 划分训练集和测试集
train_size = int(len(df_downsampled) * 0.8)
train = df_downsampled.iloc[:train_size]
test = df_downsampled.iloc[train_size:]


# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler
 
# 初始化归一化器（只fit训练集）
train_cols = train.drop(columns=['datetime']).columns
scaler = MinMaxScaler()
scaler.fit(train[train_cols])
 
# 归一化数据
train_scaled = train.copy()
test_scaled = test.copy()
train_scaled[train_cols] = scaler.transform(train[train_cols])
test_scaled[train_cols] = scaler.transform(test[train_cols])
 
# %%
# split X and y
# 以'Global_active_power'为预测目标
TARGET_COL = 'Global_active_power'
FEATURE_COLS = train_scaled.drop(columns=['datetime', TARGET_COL]).columns

def create_sequences(data, target_col, feature_cols, seq_length=24):
    X, y = [], []
    values = data[feature_cols].values
    targets = data[target_col].values
    for i in range(len(data) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(targets[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 24  # 以24小时为一个序列
X_train, y_train = create_sequences(train_scaled, TARGET_COL, FEATURE_COLS, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, TARGET_COL, FEATURE_COLS, SEQ_LENGTH)

# %%
# creat dataloaders
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# %%
# build a LSTM model
import torch.nn as nn
 
class PowerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
 
INPUT_SIZE = X_train.shape[2]  # 特征数量
model = PowerLSTM(INPUT_SIZE).to(device)
# %%
# train the model
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        batch_size = batch_x.size(0)
        train_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            batch_size = batch_x.size(0)
            test_loss += loss.item() * batch_size
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_test_loss = test_loss / len(test_loader.dataset)
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}', flush=True)
# %%
# evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()
    true_values = y_test_tensor.cpu().numpy()

# 反归一化数据（无需条件判断，直接使用虚拟特征方案）
dummy_features = np.zeros((predictions.shape[0], 6))
predictions_full = np.hstack([dummy_features, predictions.reshape(-1, 1)])
true_values_full = np.hstack([dummy_features, true_values.reshape(-1, 1)])

predictions_inverted = scaler.inverse_transform(predictions_full)[:, -1]
true_values_inverted = scaler.inverse_transform(true_values_full)[:, -1]

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(true_values_inverted, label='Predicted value', alpha=0.7)
plt.plot(predictions_inverted, label='Actual value', alpha=0.7)
plt.title('Predicted value  vs Actual value')
plt.xlabel('Time')
plt.ylabel('Power_Consumption (kW)')
plt.grid(True)
plt.legend()
plt.show()