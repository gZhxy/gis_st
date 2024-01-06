# 基于LSTM进行插值
from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
import math
from scipy.spatial.distance import squareform, pdist
import json
import geopandas as gpd
import plotnine
from plotnine import *

# 数据标准化
def transform(dt):
    return (dt - dt.min()) / (dt.max() - dt.min())
def rev_transform(dt):
    return dt * (ydata.max() - ydata.min()) + ydata.min()
# 加载数据
# 时空点样本 x,y,t,pm2.5
filename = 'data_1.csv'
data = pd.read_csv(filename, encoding='gbk')

result = {'coordinates': [], 'values': []}
# 经纬度坐标转换
def coor_convert(lon, lat):
    return [round((lon - 115) * 85, 4), round((lat - 34) * 110, 4)]

# 数据预处理
pre_station_code = ''

for _, row in data.iterrows():
    if row['station_code'] != pre_station_code:
        pre_station_code = row['station_code']
        result['coordinates'].append(coor_convert(row['lon'], row['lat']))
        result['values'].append([row['pm2_5_24h']])
        # result['time'].append()
    else:
        result['values'][-1].append(row['pm2_5_24h'])

with open('result.json', 'w') as f:
    json.dump(result, f)

coords = np.array(result['coordinates'])
vals = np.array(result['values'])

_data = np.transpose(vals) # 92*91
# 标准化
#_data = transform(_data)
_nloc = coords.shape[0]

_loc = coords
_tloc = coords

# 标准化
_loc[:, 0] = transform(_loc[:, 0])
_loc[:, 1] = transform(_loc[:, 1])
_tloc[:, 0] = transform(_tloc[:, 0])
_tloc[:, 1] = transform(_tloc[:, 1])
_nwindow = 3
_K = 10

#RollingWindow
sampl = []
nx, ny = _data.shape
for rw in range(_nwindow, nx + 1):
    si = rw - _nwindow
    series = np.transpose(_data[si:rw, :])  # nloc, nwindow
    sampl.append(series)
#print(sampl)
xdata = np.array(sampl).reshape(-1, _nloc, _nwindow)  # nsample, nloc, nwindow
#print(xdata,xdata.shape)  # (90,91,3)
_xdata = np.transpose(xdata, [0, 2, 1])  # nsample, nwindow, nloc
#print(_xdata,_xdata.shape)  # (90,3,91)
_ydata = np.transpose(_data[_nwindow - 1:, :])  # nloc(target), nsample
#print(_ydata,_ydata.shape)  # (91,90)
#_ydata = transform(_ydata0)
_xdata = _xdata[3,:,:] # (3,91)
y_true = _ydata[:,3]
print(y_true)
# GeoLayer
ntarget = _tloc.shape[0]
# R = np.zeros((ntarget,self._nloc,2))
# D = np.zeros((ntarget,self._nloc))
dset = []
dset2 = []
for i in range(ntarget):
    RLatLng = abs(_tloc[i] - _loc[:])
    #RLatLng = _tloc[i] - _loc[:]
    Dist = RLatLng[:, 0] ** 2. + RLatLng[:, 1] ** 2.
    # 剔除原站点
    Dist = np.where(Dist == 0, np.inf, Dist)
    # 排序
    idx = np.argpartition(Dist, _K)[:_K]
    # 取最近的若干站点
    R_A = RLatLng[idx, :]
    # print('0:',R_A.shape)

    xdata = _xdata[:, idx]  # nsample, nwindow, K (3,10)
    # print('1:',xdata.shape)

    # 目标站点周围站点值
    ydata0 = xdata.reshape((-1, 1))
    dset2.append(ydata0)

    R_A = np.broadcast_to(R_A, (1, _nwindow) + R_A.shape)
    # print('2:',R_A.shape)  # (1,3,10,2)
    #R_A.flags.writeable = True
    #R_A[:, :, :, 0] = transform(R_A[:, :, :, 0])
    #R_A[:, :, :, 1] = transform(R_A[:, :, :, 1])

    R_A = R_A.reshape((1, _nwindow, -1))
    # print('3:',R_A.shape)  #(1,3,20)
    dset.append(np.concatenate((xdata.reshape(1,3,10), R_A), axis=2))  # nsample, nwindow, nfeature(K x 3[RLat,Rlng,S_t)
    # print('4:',np.concatenate((xdata, R_A),axis=2).shape)  #(1,3,30)
    # update xdata
_xdata = np.array(dset)  # ntarget, nsample, nwindow, nfeature(K x 3)
print('5:',_xdata.shape)  #(91,1,3,30)
ydata = np.array(dset2)
_xdata[:,:,:,0] = transform(_xdata[:,:,:,0])
_xdata[:,:,:,1] = transform(_xdata[:,:,:,1])
_xdata[:,:,:,2] = transform(_xdata[:,:,:,2])
_xdata[:,:,:,3] = transform(_xdata[:,:,:,3])
_xdata[:,:,:,4] = transform(_xdata[:,:,:,4])
_xdata[:,:,:,5] = transform(_xdata[:,:,:,5])
_xdata[:,:,:,6] = transform(_xdata[:,:,:,6])
_xdata[:,:,:,7] = transform(_xdata[:,:,:,7])
_xdata[:,:,:,8] = transform(_xdata[:,:,:,8])
_xdata[:,:,:,9] = transform(_xdata[:,:,:,9])
input = torch.tensor(_xdata, dtype=torch.float32)
#y_stat = torch.tensor(target, dtype=torch.float32)
#print(input.shape)

class SelfAttention(nn.Module):
    def __init__(self, input_hidden_dim):
        super().__init__()
        self.hidden_dim = input_hidden_dim
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, encode_output):
        # (B, L, H) -> (B, L, 1)
        energy = self.fc(encode_output)
        weights = F.softmax(energy.squeeze(-1), dim=1)

        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encode_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class LSTM_STK(nn.Module):
    def __init__(self, input_size=30, output_size=1):
        super(LSTM_STK, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # HyperParameter
        self.hidden = 128
        self.nlayer = 4
        self.batch_size = 150

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden,
                            num_layers=self.nlayer, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(input_hidden_dim=self.hidden)
        self.fc = nn.Linear(self.hidden, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (91*90, 3, 30)
        # batch_size, seq_len = x.shape[0], x.shape[1]
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out : tensor of shape (batch_size, seq_length, 128)
        # out = out.contiguous().view(x.shape[0], -1, 2 * self.hidden)
        print(out.shape)  # 6120,3,128
        output = out[:, :, :self.hidden] + out[:, :, self.hidden:]
        weighted_out, weights = self.attention(output)
        out = self.fc(weighted_out)
        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        print(out.shape)  # 6120,1
        out = self.sigmoid(out)
        return out
'''
class LSTM_STK(nn.Module):
    def __init__(self, input_size=30, output_size=1):
        super(LSTM_STK, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # HyperParameter
        self.hidden = 128
        self.nlayer = 4
        self.batch_size = 150

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden,
                            num_layers=self.nlayer, batch_first=True,bidirectional=True)

        self.fc = nn.Linear(2 * self.hidden, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (91*90, 3, 30)
        #batch_size, seq_len = x.shape[0], x.shape[1]
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out : tensor of shape (batch_size, seq_length, 128)
        #out = out.contiguous().view(x.shape[0], -1, self.hidden)
        print(out.shape)  # 6120,3,128
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        print(out.shape)  # 6120,1
        out = self.sigmoid(out)
        return out
'''
# load
model = torch.load('lstm3.pt')
model.eval()


with torch.no_grad():
    y_pred = model.forward(input.reshape(-1,3,30))
    y_hat = rev_transform(y_pred)
    y_hat = y_hat.numpy().flatten()

    _y = abs(y_hat-y_true)
    R2 = r2_score(y_true, y_hat)
    mse = MSE(y_true, y_hat)

    rsum = np.nansum(np.fromiter(
        map(lambda x: x ** 2, _y), float)
    )
    msum = np.nansum(np.fromiter(
        map(lambda x: x / len(_y), _y), float))
    rmse = np.sqrt(rsum / len(_y))

    MAPE = np.mean(np.abs((y_true - y_hat) / y_true)) * 100
print(y_hat)
print("r2 = ", R2, "rmse = ", rmse, "mse = ", mse,"mae = ", msum, "MAPE = ", MAPE)

r = len(input.reshape(-1,3,30)) + 1
# print(y_test)

import matplotlib as mpl
from pylab import mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif']=['FangSong'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlabel('站点号',fontsize=20) # 画出坐标轴
ax.set_ylabel('PM2.5浓度',fontsize=20)
plt.xticks(list(range(0,92,5)))  # 显示的x轴刻度值
ax.tick_params(labelsize=16)
ax.plot(np.arange(1,r), y_hat, 'go-', label="预测值")
ax.plot(np.arange(1,r), y_true, 'co-', label="真实值")

plt.legend(fontsize=18)
plt.show()

# 1:1比例线
pred_min, pred_max = y_hat.min(),y_hat.max()
true_min, true_max = y_true.min(),y_true.max()

xy_mse = np.sum((y_hat-y_true)**2)
xy_mean = np.mean(y_hat)
xx_mean = np.sum((y_true - xy_mean)**2)
R2 = 1 - xy_mse/xx_mean
RMSE = np.sqrt(np.mean((y_hat - y_true)**2))

p = np.polyfit(y_hat, y_true, 1)

formatSpec = 'y = %.4fx+ %.4f'%(p[0], p[1])
formatXy = 'y = x'

x1 = np.linspace(pred_min, pred_max)
y1 = np.polyval(p, x1)

#str_R2 = '$R^2$ = %.4f\nRMSE = %.4f' % (R2, RMSE)
str_R2 = '$R^2$ = %.4f' % (R2)
f, ax = plt.subplots(figsize=(14, 10))

# plt.title('%s'%file[:-4])
#plt.title('map')
ax.tick_params(labelsize=16)  #

## cmap='BrBG'  'RdBu'
ax.scatter(x=y_hat, y=y_true, alpha=0.8, color='b')

ax.plot(np.arange(pred_min, pred_max, 0.1), np.arange(pred_min, pred_max, 0.1), color='r', linewidth=3, alpha=0.6, label=formatXy)
ax.plot(x1, y1, color='k', linewidth=3, alpha=0.6, label=formatSpec)
ax.legend(loc='lower right', fontsize=20)
# 添加文本描述
x_pos1 = int(pred_min)
y_pos1 = int(0.9 * true_max)
ax.text(x_pos1, y_pos1, str_R2, fontsize=20)
file = 'temp'
f.savefig('%s.png' % file, dpi=300, bbox_inches='tight')
plt.xlabel('预测值',fontsize=20)
plt.ylabel('真实值',fontsize=20)
plt.show()

#真实与预测误差叠加至山东地图
# 加载山东地图，获取范围
js = gpd.read_file('shandong.json')
js_box = js.geometry.total_bounds
x_s = np.linspace(js_box[0], js_box[2], 200)
y_s = np.linspace(js_box[1], js_box[3], 200)

fig, ax = plt.subplots()
js.plot(ax=ax, color="#4C92C3", alpha=0.8)
plt.show()

diff = abs(y_true-y_hat)
print(diff)

filename='经纬度.csv'
#names=['province','city','city_code','station','station_code','lon','lat','pm2_5_24h','pubtime']
data=pd.read_csv(filename,encoding='gbk')
lat1=data['lat'].values
lon1=data['lon'].values
print(lat1)
plotnine.options.figure_size = (6, 4)
idw_scatter = (ggplot() +
           geom_map(js,fill='none',color='gray',size=0.6) +
           geom_point(data,aes(x='lon1',y='lat1',fill='diff'),size=5) +
           scale_fill_cmap(cmap_name='Spectral_r',name='diff',limits = (0, 50),
                           breaks=[5,10,20,30,40,50]  )+
           scale_x_continuous(breaks=[116,118,120,122])+
           #添加文本信息
           theme(
               text=element_text(family = "SimHei"),
               #修改背景
               panel_background=element_blank(),
               axis_ticks_major_x=element_blank(),
               axis_ticks_major_y=element_blank(),
               axis_text=element_text(size=12),
               axis_title = element_text(size=14,weight="bold"),
               panel_grid_major_x=element_line(color="gray",size=.5),
               panel_grid_major_y=element_line(color="gray",size=.5),
            ))
idw_scatter.save('diff_11.6.jpg',dpi=300)