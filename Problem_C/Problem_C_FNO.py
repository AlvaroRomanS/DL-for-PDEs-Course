import numpy as np
import h5py
import torch 
from scipy.interpolate import griddata
device = 'cuda'
dtype = torch.float32



import matplotlib.font_manager as fm
import matplotlib as mpl

# font setting for figures
matches = [f.name for f in fm.fontManager.ttflist if 'CMU' in f.name]
print(matches)  # should include 'CMU Sans Serif'
print(mpl.get_cachedir())


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['CMU Sans Serif']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams['mathtext.rm'] = 'CMU Sans Serif'
mpl.rcParams['mathtext.it'] = 'CMU Sans Serif:italic'
mpl.rcParams['mathtext.bf'] = 'CMU Sans Serif:bold'
#mpl.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt



print('START')

######################################
# Load training data
######################################


# data import
with h5py.File('Problem_C/ProblemC_dataset.h5', 'r') as f:
    print('The dataset for Problem C:', list(f.keys()))
    a_train = np.array(f['a_train'])
    u_train = np.array(f['u_train'])
    a_test = np.array(f['a_test'])
    u_test = np.array(f['u_test'])
    X = np.array(f['X'])
    Y = np.array(f['Y'])

res = X.shape[0]  # 128

# coordsinates [128, 128, 2] -> [1, 128, 128, 2]
coords = np.stack([X, Y], axis=-1)
gridx = coords.reshape(-1, 2)
x     = torch.tensor(coords, dtype=dtype).unsqueeze(0)

def transform_data(a_np, u_np, x_coords):
    # this extends the tensor a(x) into the time dimension
    N = a_np.shape[0]
    a = torch.tensor(a_np, dtype=dtype).unsqueeze(-1)
    u = torch.tensor(u_np, dtype=dtype).unsqueeze(-1)
    ax = torch.cat([a, x_coords.repeat(N, 1, 1, 1)], dim=-1)
    return ax, u

ax_train, u_train = transform_data(a_train, u_train, x)
ax_test,  u_test = transform_data(a_test,  u_test,  x)

print('ax_train:', ax_train.shape)
print('u_train: ', u_train.shape)
print('ax_test: ', ax_test.shape)
print('u_test:  ', u_test.shape)

####### Visualize the training data
# 3 examples to visualise
indices = [0, 5, 7]
#
mesh = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 200))
x_plot, y_plot = mesh[0], mesh[1]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 7))
#
for row, idx in enumerate(indices):
    a_show = ax_train[idx, ..., 0]
    u_show = u_train[idx]

    z_a = griddata((gridx[:,0], gridx[:,1]), np.ravel(a_show), (x_plot, y_plot), method='cubic')
    cntr0 = axs[row, 0].contourf(x_plot, y_plot, z_a, levels=40, cmap='jet')
    fig.colorbar(cntr0, ax=axs[row, 0])
    axs[row, 0].set_title(f'True a (sample {idx})')
    axs[row, 0].set_xlabel('x')
    axs[row, 0].set_ylabel('y')

    z_u = griddata((gridx[:,0], gridx[:,1]), np.ravel(u_show), (x_plot, y_plot), method='cubic')
    cntr1 = axs[row, 1].contourf(x_plot, y_plot, z_u, levels=40, cmap='jet')
    fig.colorbar(cntr1, ax=axs[row, 1])
    axs[row, 1].set_title(f'True u (sample {idx})')
    axs[row, 1].set_xlabel('x')
    axs[row, 1].set_ylabel('y')

plt.tight_layout()
plt.show()
plt.close()




import torch.nn as nn

class SpectralConv2d(nn.Module):
    
    def __init__(self, in_size, out_size, modes1, modes2, dtype):
        super(SpectralConv2d, self).__init__()
        '''2D Fourier layer: FFT -> linear transform -> Inverse FFT
        '''
        self.in_size = in_size 
        self.out_size = out_size 
        self.modes1 = modes1
        self.modes2 = modes2 
        #
        self.scale = 1./(in_size * out_size)
        #
        if (dtype is None) or (dtype==torch.float32):
            ctype = torch.complex64 
        elif (dtype==torch.float64):
            ctype = torch.complex128 
        else:
            raise TypeError(f'No such data type.')
        #
        self.weight1 = nn.Parameter(self.scale * torch.rand(in_size, out_size, 
                                                            modes1, modes2, 
                                                            dtype=ctype))
        self.weight2 = nn.Parameter(self.scale * torch.rand(in_size, out_size, 
                                                            modes1, modes2, 
                                                            dtype=ctype))
    
    def compl_mul_2d(self, input, weights):
        '''Complex multiplication: (batch_size, in_size, m1, m2) * (in_size, out_size, m1, m2) -> (batch_size, out_size, m1, m2)
        '''
        return torch.einsum('bixy,ioxy->boxy', input, weights)

    def forward(self, x):
        '''
        Input:
            x: size(batch_size, in_size, my_size, mx_size)
        Return:
            x: size(batch_size, out_size, my_size, mx_size)
        '''
        batch_size = x.shape[0]
        ####################### Compute Fourier coefficients up to a factor of e^{-c}
        x_ft = torch.fft.rfft2(x) # size(batch_size, in_size, mx_size, my_size//2+1)
        ######################## Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_size, x.size(-2), x.size(-1)//2+1, 
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] \
           = self.compl_mul_2d(x_ft[:, :, :self.modes1, :self.modes2], self.weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] \
           = self.compl_mul_2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weight2)
        ######################### Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # size(batch_size, out_size, mx_size, my_size)

        return x 
        
class FNO2d(nn.Module):

    def __init__(self, in_size:int, out_size:int, modes1:int, modes2:int, 
                 hidden_list:list[int], dtype=None):
        super(FNO2d, self).__init__()
        self.hidden_list = hidden_list
        # Activation
        self.activation = nn.ReLU()
        # The input layer: LIFTING STEP (usually called P)
        self.fc_in = nn.Linear(in_size, hidden_list[0], dtype=dtype)
        # The hidden layer
        conv_net, w_net = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list[1:]:
            conv_net.append(SpectralConv2d(self.hidden_in, hidden, modes1, modes2, dtype))
            w_net.append(nn.Conv1d(self.hidden_in, hidden, 1, dtype=dtype))
            self.hidden_in =  hidden 
        self.spectral_conv = nn.Sequential(*conv_net)
        self.weight_conv = nn.Sequential(*w_net)
        # The output layer: PROJECTION STEP (usually called Q)
        self.fc_out0 = nn.Linear(self.hidden_in, 128, dtype=dtype)
        self.fc_out1 = nn.Linear(128, out_size, dtype=dtype)
    
    def forward(self, ax):
        '''
        Input: 
            ax: size(batch_size, my_size, mx_size, a_size+x_size)
        Output: 
            u: size(batch_size, my_size, mx_size, out_size)
        '''
        batch_size = ax.shape[0]
        mx_size, my_size = ax.shape[1], ax.shape[2]
        # The input layer: size(b, mx_size, my_size, in_size) -> (b, hidden_size, my_size, mx_size)
        ax = self.fc_in(ax) # LIFTING STEP
        ax = ax.permute(0, 3, 1, 2)
        # The spectral conv layer (FOURIER LAYER)
        hidden_last = self.hidden_list[0] 
        for conv, weight, hidden_size in zip(self.spectral_conv, self.weight_conv, self.hidden_list[1:]):
            ax1 = conv(ax)   # size(b, hidden_size, my_size, mx_size)
            ax2 = weight(ax.view(batch_size, hidden_last, -1)).view(batch_size, hidden_size, mx_size, my_size)
            ax = self.activation(ax1+ax2)
            hidden_last = hidden_size 
        # The output layer: size(batch_size, hidden_size, my_size, mx_size) -> size(batch_size, my_size, mx_size, out_size)
        ax = ax.permute(0, 2, 3, 1)
        ax = self.fc_out0(ax) # PROJECTION STEP
        ax = self.activation(ax) # PROJECTION STEP

        return self.fc_out1(ax)
#
mode1, mode2 = 16, 16
hidden_list = [40, 40, 40] # lifting layer, Fourier layer 1, fourier layer 2
model_u = FNO2d(ax_train.shape[-1], u_train.shape[-1], mode1, mode2, hidden_list).to(device)
################## Find total trainable parameters
total_trainable_params = sum(p.numel() for p in model_u.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')




class UnitGaussianNormalizer():

    def __init__(self, ax, eps=1e-8):
        super(UnitGaussianNormalizer, self).__init__()
        '''Apply normaliztion to the first dimension of last axis of ax
        Input:
            ax: size(N, mesh_size, 1+d)
        Output:
            mean: size(mesh_szie, 1)
            std: size(mesh_size, 1)
        '''
        self.mean = torch.mean(ax[...,0:1], 0)
        self.std = torch.std(ax[...,0:1], 0)
        self.eps = eps
    
    def encode(self, ax):
        '''
        Input:
            ax: ax: size(N, mesh_size, 1+d)
        '''
        d = ax.shape[-1] - 1
        ax_list = torch.split(ax, split_size_or_sections=[1, d], dim=-1)
        ax = torch.cat([(ax_list[0]-self.mean) / (self.std + self.eps), ax_list[1]], dim=-1)

        return ax
    
    def decode(self, ax):
        #
        d = ax.shape[-1] - 1
        ax_list = torch.split(ax, split_size_or_sections=[1, d], dim=-1)
        ax = torch.cat([ax_list[0] * (self.std + self.eps) + self.mean, ax_list[1]], dim=-1)

        return ax
#
normalizer_ax = UnitGaussianNormalizer(ax_train.to(device))
normalizer_u = UnitGaussianNormalizer(u_train.to(device))

# The loss function
class LossClass(object):

    def __init__(self, u_model):
        self.device = device
        self.u_model = u_model 
    
    def loss_data(self, ax_batch, u_batch):
        '''loss term'''
        batch_size = u_batch.shape[0]
        ax, u = ax_batch.to(self.device), u_batch.to(self.device)
        # with normalizer
        ax_norm = normalizer_ax.encode(ax)
        u_pred_norm = self.u_model(ax_norm)
        u_pred = normalizer_u.decode(u_pred_norm)
        # w/o normalizer
        # u_pred = self.u_model(ax)
        loss = torch.norm(u.reshape(batch_size, -1)-u_pred.reshape(batch_size, -1), 2, 1)
        loss = torch.mean(loss)
        
        return loss 
    
    def get_error(self, ax, u):
        '''L2 relative error'''
        batch_size = u.shape[0]
        ax, u = ax.to(self.device), u.to(self.device)
        # with normalizer
        ax_norm = normalizer_ax.encode(ax)
        u_pred_norm = self.u_model(ax_norm)
        u_pred = normalizer_u.decode(u_pred_norm)
        # w/o normalizer
        # u_pred = self.u_model(ax)
        error = torch.norm(u.reshape(batch_size,-1)-u_pred.reshape(batch_size,-1), 2, 1) / torch.norm(u.reshape(batch_size,-1), 2, 1)

        return torch.mean(error)
    


from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import time
############# Define your own dataset 
class MyDataset(Dataset):

    def __init__(self, ax:torch.tensor, u:torch.tensor):
        '''
        Input:
            ax: size(batch_size, a_size+x_size)
            u: size(batch_size, u_size)
        '''
        self.ax = ax
        self.u = u
    
    def __getitem__(self, index):
        return self.ax[index], self.u[index]

    def __len__(self):
        return self.ax.shape[0]

train_loader = DataLoader(MyDataset(ax_train, u_train), batch_size=5, shuffle=True)

############ Training setups
epochs = 30 # with 30 epochs the L^2 is still decreasing, can use more epochs on final training run
lr = 1e-3
optimizer = torch.optim.Adam(params=model_u.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.5)
loss_list, error_list = [], []
t0 = time.time()
for epoch in trange(epochs):
    loss = 0.
    for ax_batch, u_batch in train_loader:
        lossClass = LossClass(model_u)
        loss_train = lossClass.loss_data(ax_batch, u_batch)
        #
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        #
        loss += loss_train
    #
    scheduler.step()  # Adjust learning rate
    with torch.no_grad():
        error = lossClass.get_error(ax_test, u_test)
        error_list.append(error.item())
    #
    loss = loss/len(train_loader)
    loss_list.append(loss.item())
    if (epoch+1)%100==0:
        print(f'Epoch:{epoch}, The loss is:{loss.item()}')
        print('error_test:', error_list.pop())
print('The consuming time is:', time.time()-t0)




#######################################
# The L2 relative error
#######################################
def L2_error(u, u_pred, ndata=200):
    ''' '''
    l2 = torch.norm(u.reshape(ndata,-1)-u_pred.reshape(ndata,-1), 2, 1) / torch.norm(u.reshape(ndata, -1), 2, 1)
    return l2
# with normalizer
ax_test_norm = normalizer_ax.encode(ax_test.to(device))
u_test_pred_norm = model_u(ax_test_norm)
u_test_pred = normalizer_u.decode(u_test_pred_norm).detach().cpu()
# w/o normalizer
# u_test_pred = model_u(ax_test.to(device)).detach().cpu()
print('The shape of u_pred:', u_test_pred.shape)
l2_err = L2_error(u_test, u_test_pred)
print('The average l2 error:', torch.mean(l2_err))

#######################################
# Visualize the prediction and truth u
#######################################
indices = [0, 5, 7] #visualization of different test instances
#
mesh = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 200))
x_plot, y_plot = mesh[0], mesh[1]
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 7))
#
for row, idx in enumerate(indices):
    u_true = u_test[idx]
    u_pred = u_test_pred[idx]

    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(u_true), (x_plot, y_plot), method='cubic')
    cntr0 = axs[row, 0].contourf(x_plot, y_plot, z_plot, levels=40, cmap='jet')
    fig.colorbar(cntr0, ax=axs[row, 0])
    axs[row, 0].set_title(f'True u (sample {idx})')
    axs[row, 0].set_xlabel('x')
    axs[row, 0].set_ylabel('y')

    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(u_pred), (x_plot, y_plot), method='cubic')
    cntr1 = axs[row, 1].contourf(x_plot, y_plot, z_plot, levels=40, cmap='jet')
    fig.colorbar(cntr1, ax=axs[row, 1])
    axs[row, 1].set_title(f'Pred. u (sample {idx})')
    axs[row, 1].set_xlabel('x')
    axs[row, 1].set_ylabel('y')

    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(np.abs(u_true - u_pred)), (x_plot, y_plot), method='cubic')
    cntr2 = axs[row, 2].contourf(x_plot, y_plot, z_plot, levels=40, cmap='jet')
    fig.colorbar(cntr2, ax=axs[row, 2])
    axs[row, 2].set_title(f'Absolute error (sample {idx})')
    axs[row, 2].set_xlabel('x')
    axs[row, 2].set_ylabel('y')

plt.tight_layout()
plt.show()
#############################




fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(error_list)
ax.set_title(f'Final $L^2$ error: $u = {error_list[-1]:.4f}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('$L^2$ relative error')
plt.tight_layout()
fig.savefig('Problem_C/error_vs_epoch.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


####################################################



#######################################
# Visualize the prediction and truth u
#######################################
indices = [0]

mesh = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 200))
x_plot, y_plot = mesh[0], mesh[1]

for idx in indices:
    u_true = u_test[idx]
    u_pred = u_test_pred[idx]
    u_err = np.abs(u_true - u_pred)
    a_true = a_test[idx]

    vmin_u = min(u_true.min(), u_pred.min())
    vmax_u = max(u_true.max(), u_pred.max())

    # ground truth u(x)
    fig, ax = plt.subplots(figsize=(5, 5))
    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(u_true), (x_plot, y_plot), method='cubic')
    cntr = ax.contourf(x_plot, y_plot, z_plot, levels=70, cmap='jet', vmin=vmin_u, vmax=vmax_u)
    ax.set_box_aspect(1)
    fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
    ax.set_title(f'Ground truth $u(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.tight_layout()
    fig.savefig('Problem_C/GT_u.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    #Prediction u(x)
    fig, ax = plt.subplots(figsize=(5, 5))
    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(u_pred), (x_plot, y_plot), method='cubic')
    cntr = ax.contourf(x_plot, y_plot, z_plot, levels=70, cmap='jet', vmin=vmin_u, vmax=vmax_u)
    ax.set_box_aspect(1)
    fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
    ax.set_title(f'Prediction $u_{{\\theta}}(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.tight_layout()
    fig.savefig('Problem_C/predicted_u.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Material field a(x)
    fig, ax = plt.subplots(figsize=(5, 5))
    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(a_true), (x_plot, y_plot), method='cubic')
    cntr = ax.contourf(x_plot, y_plot, z_plot, levels=70, cmap='jet')
    ax.set_box_aspect(1)
    fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
    ax.set_title(f'Material field $a(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.tight_layout()
    fig.savefig('Problem_C/input_a.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Pointwise error
    fig, ax = plt.subplots(figsize=(5, 5))
    z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(u_err), (x_plot, y_plot), method='cubic')
    cntr = ax.contourf(x_plot, y_plot, z_plot, levels=70, cmap='jet')
    ax.set_box_aspect(1)
    fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
    ax.set_title(f'Pointwise error $|u(x) - u_{{\\theta}}(x)|$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.tight_layout()
    fig.savefig('Problem_C/pointwise_error_u.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


print('END')