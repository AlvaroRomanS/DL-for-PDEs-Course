import numpy as np
import h5py
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from tqdm import trange
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3407)
device = 'cuda'
dtype = torch.float32



import matplotlib.font_manager as fm
import matplotlib as mpl

# font settings
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
t_start = time.time()








##########################################################################3





with h5py.File('Problem_D/ProblemD_dataset.h5', 'r') as f:
    print('The dataset for Problem D:', list(f.keys()))
    a_train_labeled = np.array(f['a_train_labeled'])
    u_train_labeled = np.array(f['u_train_labeled'])
    a_train_unlabeled = np.array(f['a_train_unlabeled'])
    a_test = np.array(f['a_test'])
    u_test = np.array(f['u_test'])
    x_mesh = np.array(f['x_mesh'])
    t_mesh = np.array(f['t_mesh'])

N_t, N_x = t_mesh.shape[0], x_mesh.shape[0]  # 200, 256


t_grid, x_grid = np.meshgrid(t_mesh.ravel(), x_mesh.ravel(), indexing='ij')
gridx = np.stack([t_grid, x_grid], axis=-1).reshape(-1, 2)

def transform_data(a_np, x_mesh, t_mesh, u_np=None):
    N = a_np.shape[0]
    # broadcast a across the time dimension
    a = torch.tensor(a_np, dtype=dtype).view(N, 1, N_x, 1).repeat(1, N_t, 1, 1)

    t_coord = torch.tensor(t_mesh, dtype=dtype).view(1, N_t, 1, 1).repeat(N, 1, N_x, 1)
    x_coord = torch.tensor(x_mesh, dtype=dtype).view(1, 1, N_x, 1).repeat(N, N_t, 1, 1)
    ax = torch.cat([a, t_coord, x_coord], dim=-1)
    u = None
    if u_np is not None:
        u = torch.tensor(u_np, dtype=dtype).unsqueeze(-1)

    return ax, u

ax_train_labeled, u_train_labeled = transform_data(a_train_labeled, x_mesh, t_mesh, u_train_labeled)
ax_train_unlabeled, _ = transform_data(a_train_unlabeled, x_mesh, t_mesh, u_np=None)
ax_test, u_test = transform_data(a_test, x_mesh, t_mesh, u_test)

print('ax_train_labeled:', ax_train_labeled.shape)
print('u_train_labeled:', u_train_labeled.shape)
print('ax_train_unlabeled:', ax_train_unlabeled.shape)
print('ax_test:', ax_test.shape)
print('u_test:', u_test.shape)
print('gridx:', gridx.shape)

##########################################################################






######################################
# Visualize one training sample
######################################
a_show = ax_train_labeled[0,...,0]
mesh = np.meshgrid(np.linspace(0, 1, 200), np.linspace(-1, 1, 256))
x_plot, y_plot = mesh[0], mesh[1]

fig, ax = plt.subplots(figsize=(6, 4))
print('MY CODE:')
print(f'gridx shape = {gridx.shape}')
print(f'a_show shape = {a_show.shape}')
print(f'x_plot shape = {x_plot.shape}')
print(f'y_plot shape = {y_plot.shape}')
z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(a_show), (x_plot, y_plot), method='cubic')
cntr = ax.contourf(x_plot, y_plot, z_plot, levels=40, cmap='jet')
fig.colorbar(cntr, ax=ax)
ax.set_title('Input coefficient a (training, unlabeled)')
ax.set_xlabel('t'); ax.set_ylabel('x')
#plt.show()
plt.close()







class SpectralConv2d(nn.Module):
    def __init__(self, in_size, out_size, modes1, modes2, dtype):
        super(SpectralConv2d, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1. / (in_size * out_size)
        ctype = torch.complex64 if (dtype is None or dtype == torch.float32) else torch.complex128
        self.weight1 = nn.Parameter(self.scale * torch.rand(in_size, out_size, modes1, modes2, dtype=ctype))
        self.weight2 = nn.Parameter(self.scale * torch.rand(in_size, out_size, modes1, modes2, dtype=ctype))

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
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch_size, self.out_size, x.size(-2), x.size(-1)//2+1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul_2d(x_ft[:, :, :self.modes1, :self.modes2], self.weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul_2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weight2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, in_size, out_size, modes1, modes2, hidden_list, dtype=None):
        super(FNO2d, self).__init__()
        self.hidden_list = hidden_list
        # Activation
        self.activation = nn.ReLU()
        # The input layer
        self.fc_in = nn.Linear(in_size, hidden_list[0], dtype=dtype)
        # The hidden layers
        conv_net, w_net = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list[1:]:
            conv_net.append(SpectralConv2d(self.hidden_in, hidden, modes1, modes2, dtype))
            w_net.append(nn.Conv1d(self.hidden_in, hidden, 1, dtype=dtype))
            self.hidden_in = hidden
        self.spectral_conv = nn.ModuleList(conv_net)   # ModuleList for proper registration
        self.weight_conv = nn.ModuleList(w_net)
        # The output layer
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
        # The input layer
        ax = self.fc_in(ax).permute(0, 3, 1, 2)
        # The spectral conv layer 
        hidden_last = self.hidden_list[0]
        for conv, weight, hidden_size in zip(self.spectral_conv, self.weight_conv, self.hidden_list[1:]):
            ax1 = conv(ax)
            ax2 = weight(ax.view(batch_size, hidden_last, -1)).view(batch_size, hidden_size, mx_size, my_size)
            ax = self.activation(ax1 + ax2)
            hidden_last = hidden_size
        # The output layer
        ax = ax.permute(0, 2, 3, 1)
        ax = self.activation(self.fc_out0(ax))
        return self.fc_out1(ax)


mode1, mode2 = 8, 8
hidden_list = [20, 20, 20]
model_u = FNO2d(ax_train_labeled.shape[-1], u_train_labeled.shape[-1],
                      mode1, mode2, hidden_list).to(device)
################## Find total trainable parameters
total_params = sum(p.numel() for p in model_u.parameters() if p.requires_grad)
print(f'{total_params:,} trainable parameters.')




######################################
# Normaliser 

class UnitGaussianNormalizer:
    def __init__(self, ax, eps=1e-8):
        self.mean = torch.mean(ax[..., 0:1], 0)
        self.std = torch.std( ax[..., 0:1], 0)
        self.eps = eps

    def encode(self, ax):
        d = ax.shape[-1] - 1
        a_norm, x_part = torch.split(ax, [1, d], dim=-1)
        return torch.cat([(a_norm - self.mean) / (self.std + self.eps), x_part], dim=-1)

    def decode(self, ax):
        d = ax.shape[-1] - 1
        a_norm, x_part = torch.split(ax, [1, d], dim=-1)
        return torch.cat([a_norm * (self.std + self.eps) + self.mean, x_part], dim=-1)

normalizer_ax = UnitGaussianNormalizer(ax_train_labeled.to(device))

# #####################################
class mollifier(object):

    def __inint__(self):
        super(mollifier, self).__init_()
        
    def __call__(self, u_tilde, ax):

        a = ax[..., 0:1]
        t = ax[..., 1:2]
        x = ax[..., 2:3]

        spatial_mask = torch.sin(np.pi * (x + 1) / 2)   # vanishes at x = ±1
        u = a + t * spatial_mask * u_tilde
        return u

######################################

nu = 1.0/np.pi

class LossClass(object):

    def __init__(self, u_model, lambda_r=1.0, lambda_d=0.0):
        """
        Args:
            u_model : FNO2d
            lambda_r : weight for PDE residual loss
            lambda_d : weight for supervised data loss (0 = fully PI)
        """
        self.u_model = u_model
        self.mollifier = mollifier()
        self.lambda_r = lambda_r
        self.lambda_d = lambda_d
        #
        self.dt = 1/(N_t-1)
        self.dx = 1/(N_x-1)

    def _predict(self, ax_batch):
        """Forward pass with normalisation and ansatz. Returns u: (B, N_t, N_x)."""
        ax_norm = normalizer_ax.encode(ax_batch)
        u_tilde = self.u_model(ax_norm)
        u = self.mollifier(u_tilde, ax_batch)
        return u[..., 0]

    def loss_pde(self, u_pred):
        n_batch = u_pred.shape[0]

        # the strong residual derivatives are approximated with forward differencing
        dudt = (u_pred[:, 2:, 1:-1] - u_pred[:, :-2, 1:-1]) / (2 * self.dt)
        dudx = (u_pred[:, 1:-1, 2:] - u_pred[:, 1:-1, :-2]) / (2 * self.dx)
        d2udx2 = (u_pred[:, 1:-1, 2:] - 2*u_pred[:, 1:-1, 1:-1] + u_pred[:, 1:-1, :-2]) / (self.dx**2)

        u_interior = u_pred[:, 1:-1, 1:-1]
        residual = dudt + u_interior * dudx - nu * d2udx2

        loss = torch.norm(residual.reshape(n_batch, -1), 2, dim=1)
        return torch.mean(loss)

    def loss_data(self, u_pred, u_batch):
        batch_size = u_batch.shape[0]
        u_true = u_batch[..., 0]
        loss = torch.norm(
            (u_true - u_pred).reshape(batch_size, -1), 2, dim=1
        )
        return torch.mean(loss)

    def loss_total(self, ax_batch, u_batch=None):
        u_pred = self._predict(ax_batch)  # single forward pass

        loss = self.lambda_r * self.loss_pde(u_pred)
        #print(f'PDE loss: {loss.item():.4e}')
        if self.lambda_d > 0 and u_batch is not None:
            data_loss = self.lambda_d * self.loss_data(u_pred, u_batch)
            loss += data_loss
            #print(f'Data loss: {data_loss.item():.4e}')

        return loss

    def get_error(self, ax, u):
        """L2 relative error on labeled pairs."""
        batch_size = u.shape[0]
        u_pred = self._predict(ax)
        u_true = u[..., 0]
        error = (torch.norm((u_true - u_pred).reshape(batch_size, -1), 2, dim=1)
                / torch.norm( u_true          .reshape(batch_size, -1), 2, dim=1))
        return torch.mean(error)
    





class MyDataset(Dataset):
    def __init__(self, ax: torch.Tensor, u: torch.Tensor = None):
        self.ax = ax
        self.u = u

    def __getitem__(self, index):
        if self.u is not None:
            return self.ax[index], self.u[index]
        return (self.ax[index],)

    def __len__(self):
        return self.ax.shape[0]
        

print(f'Pre-processing time:  {time.time()-t_start:.2f}s')

######################################
# Training
#
# PINO mode  (lambda_d = 0): train on PDE loss only, no labeled u.
# Semi-supervised (lambda_d > 0): add data loss on available labeled pairs.
######################################
LAMBDA_R = 1.0    # PDE residual weight
LAMBDA_D = 2000.0    # Data loss weight — set > 0 for semi-supervised


print(f'Check if tensors on GPU')
print(ax_train_labeled.device)
print(ax_train_unlabeled.device)

print(f'Tensors shape and type')
print(ax_train_labeled.shape, ax_train_labeled.dtype)
print(ax_train_unlabeled.shape, ax_train_unlabeled.dtype)


ax_train_labeled_gpu = ax_train_labeled.to(device)
u_train_labeled_gpu = u_train_labeled.to(device)
ax_train_unlabeled_gpu = ax_train_unlabeled.to(device)

labeled_dataset = MyDataset(ax_train_labeled_gpu, u_train_labeled_gpu)
unlabeled_dataset = MyDataset(ax_train_unlabeled_gpu)

batch_size = 10

# Labeled: has both a(x) and u(x): FOR LABELED DATA
##labeled_dataset = MyDataset(ax_train_labeled, u_train_labeled)
labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

# Unlabeled: only a(x), no u(x): FOR UNLABELED DATA
##unlabeled_dataset = MyDataset(ax_train_unlabeled)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

# FOR EVALUATION USING TEST SET
# IMPORTANT: Create a temporary dataset and loader to be able to pass a batch of the 
        # same size as the train_loader into the model.
        # If this is not done if causes an error in the inverse FFT since the
        # batch size is different than expected on the forward pass.
test_dataset = TensorDataset(ax_test, u_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

lossClass = LossClass(
    u_model=model_u,
    lambda_r=LAMBDA_R,
    lambda_d=LAMBDA_D)

epochs = 30 # best result i got today was with 30
lr = 5e-4
optimizer = torch.optim.Adam(model_u.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

loss_list, error_list = [], []

t_train_start = time.time()

for epoch in trange(epochs):
    model_u.train()
    epoch_loss = 0.0

    counter = 0
    t_epoch_start = time.time()
    for labeled_batch, unlabeled_batch in zip(labeled_loader, unlabeled_loader):

        #print(f'\n--- BATCH {counter} ---')
        
        t0 = time.time()
        ##ax_labeled = labeled_batch[0].to(device)
        ##u_labeled = labeled_batch[1].to(device)
        ##ax_unlabeled = unlabeled_batch[0].to(device)
        ax_labeled = labeled_batch[0]
        u_labeled = labeled_batch[1]
        ax_unlabeled = unlabeled_batch[0]
        t1 = time.time()
        #print(f'Data time:  {t1-t0:.4f}s')

        # Labeled: PDE loss + data loss
        loss_labeled = lossClass.loss_total(ax_labeled, u_labeled)
        # Unlabeled: PDE loss only (no u passed)
        loss_unlabeled = lossClass.loss_total(ax_unlabeled)

        loss_train = loss_labeled + loss_unlabeled
        t2 = time.time()
        #print(f'Forward time:  {t2-t1:.4f}s')

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        t3 = time.time()
        #print(f'Backward time:  {t3-t2:.4f}s')

        epoch_loss += loss_train.detach()
        counter += 1
        t4 = time.time()
        #print(f'Epoch loss add time:  {t4-t3:.4f}s')

    print(f'\n-------------------')
    print(f'EPOCH TIME: {time.time() - t_epoch_start:.4f}s')
    print(f'EPOCH LOSS:  {epoch_loss:.4f}')
    print(f'-------------------')

    t5 = time.time()
    scheduler.step()

    # evaluate on labeled test set (for monitoring only)
    model_u.eval()
    with torch.no_grad():
        
        errors = []
        for ax_b, u_b in test_loader:
            errors.append(lossClass.get_error(ax_b.to(device), u_b.to(device)).item())
        error = np.mean(errors)
        error_list.append(error)

    epoch_loss = epoch_loss.cpu().item() / len(labeled_loader)
    loss_list.append(epoch_loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1:4d} | PDE loss: {epoch_loss:.4e} '
              f'| Test L2 rel. error: {error_list[-1]:.4e}')
    t6 = time.time()
    print(f'Eval time:  {t6-t5:.4f}s\n')

print(f'\nTotal training time: {time.time() - t_train_start:.1f}s')






#######################################
# The L2 relative error
#######################################
def L2_error(u, u_pred):
    ''' '''
    l2 = torch.norm(u-u_pred, 2, 1) / torch.norm(u, 2, 1)
    return l2
######################################
# Visualise predictions vs ground truth (test sample)
######################################
model_u.eval()
with torch.no_grad():
    u_pred_list = []
    test_dataset = TensorDataset(ax_test)
    test_loader = DataLoader(test_dataset, batch_size=10)
    
    for (ax_b,) in test_loader:
        ax_norm = normalizer_ax.encode(ax_b.to(device))
        u_pred = model_u(ax_norm)
        u_pred = mollifier()(u_pred, ax_b.to(device)).cpu()
        u_pred_list.append(u_pred)
    
    u_pred_all = torch.cat(u_pred_list, dim=0)
# compute the L2 relative error
l2_err = L2_error(u_test.reshape(-1,N_t*N_x), u_pred_all.reshape(-1,N_t*N_x))
print('The average l2 error:', torch.mean(l2_err))

sample_idx = 0
a_show = ax_test[sample_idx, ..., 0].numpy()
u_show = u_test[ sample_idx, ..., 0].numpy()
u_pred_show = u_pred_all[sample_idx, ..., 0].numpy()

mesh_plot = np.meshgrid(np.linspace(0, 1, 200), np.linspace(-1, 1, 256))
x_plot, y_plot = mesh_plot[0], mesh_plot[1]

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

for ax, vals, title in zip(axs,
                            [a_show, u_show, u_pred_show],
                            ['Input a', 'True u', 'Predicted u (PINO)']):
    z = griddata((gridx[:,0], gridx[:,1]), vals.ravel(), (x_plot, y_plot), method='cubic')
    cntr = ax.contourf(x_plot, y_plot, z, levels=40, cmap='jet')
    fig.colorbar(cntr, ax=ax)
    ax.set_title(title); ax.set_xlabel('t'); ax.set_ylabel('x')

plt.tight_layout()
plt.show()






# Training curves

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].semilogy(loss_list);  axs[0].set_title('PDE Residual Loss'); axs[0].set_xlabel('Epoch')
axs[1].semilogy(error_list); axs[1].set_title('Test L2 Relative Error'); axs[1].set_xlabel('Epoch')
plt.tight_layout()
plt.show()
plt.close()



#############################


fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(error_list)
ax.set_title(f'Final $L^2$ error: $u = {error_list[-1]:.4f}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('$L^2$ relative error')
plt.tight_layout()
fig.savefig('Problem_D/error_vs_epoch.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



######################################
# Visualise predictions vs ground truth (test sample)
######################################

model_u.eval()
with torch.no_grad():

    u_pred_list = []

    test_dataset = TensorDataset(ax_test)
    test_loader = DataLoader(test_dataset, batch_size=10)

    for (ax_b,) in test_loader:
        ax_norm = normalizer_ax.encode(ax_b.to(device))
        u_pred = model_u(ax_norm)
        u_pred = mollifier()(u_pred, ax_b.to(device)).cpu()
        u_pred_list.append(u_pred)

    u_pred_all = torch.cat(u_pred_list, dim=0)


# compute the L2 relative error
l2_err = L2_error(
    u_test.reshape(-1, N_t*N_x),
    u_pred_all.reshape(-1, N_t*N_x)
)

print('The average l2 error:', torch.mean(l2_err))


sample_idx = 0

a_show = ax_test[sample_idx, ..., 0].numpy()
u_show = u_test[sample_idx, ..., 0].numpy()
u_pred_show = u_pred_all[sample_idx, ..., 0].numpy()

mesh_plot = np.meshgrid(
    np.linspace(0, 1, 200),
    np.linspace(-1, 1, 256)
)

x_plot, y_plot = mesh_plot


vmin_u = min(u_show.min(), u_pred_show.min())
vmax_u = max(u_show.max(), u_pred_show.max())



# material field a(x,t)


fig, ax = plt.subplots(figsize=(5,5))
z_plot = griddata(
    (gridx[:,0], gridx[:,1]),
    a_show.ravel(),
    (x_plot, y_plot),
    method='cubic'
)
cntr = ax.contourf(
    x_plot,
    y_plot,
    z_plot,
    levels=70,
    cmap='jet'
)
ax.set_box_aspect(1)
fig.colorbar(
    cntr,
    ax=ax,
    shrink=0.74,
    pad=0.04
)
ax.set_title(r'Initial velocity profile $a(x,t)$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
plt.tight_layout()
fig.savefig('Problem_D/input_a.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



# Ground truth
fig, ax = plt.subplots(figsize=(5,5))
z_plot = griddata(
    (gridx[:,0], gridx[:,1]),
    u_show.ravel(),
    (x_plot, y_plot),
    method='cubic'
)
cntr = ax.contourf(
    x_plot,
    y_plot,
    z_plot,
    levels=70,
    cmap='jet',
    vmin=vmin_u,
    vmax=vmax_u
)
ax.set_box_aspect(1)
fig.colorbar(
    cntr,
    ax=ax,
    shrink=0.74,
    pad=0.04
)
ax.set_title(r'Ground truth $u(x,t)$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
plt.tight_layout()
fig.savefig('Problem_D/GT_u.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



#prediction u_theta
fig, ax = plt.subplots(figsize=(5,5))
z_plot = griddata(
    (gridx[:,0], gridx[:,1]),
    u_pred_show.ravel(),
    (x_plot, y_plot),
    method='cubic'
)
cntr = ax.contourf(
    x_plot,
    y_plot,
    z_plot,
    levels=70,
    cmap='jet',
    vmin=vmin_u,
    vmax=vmax_u
)
ax.set_box_aspect(1)
fig.colorbar(
    cntr,
    ax=ax,
    shrink=0.74,
    pad=0.04
)
ax.set_title(r'Prediction $u_{\theta}(x,t)$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
plt.tight_layout()
fig.savefig('Problem_D/predicted_u.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



# Pointwise error

u_err = np.abs(u_show - u_pred_show)
fig, ax = plt.subplots(figsize=(5,5))
z_plot = griddata(
    (gridx[:,0], gridx[:,1]),
    u_err.ravel(),
    (x_plot, y_plot),
    method='cubic'
)
cntr = ax.contourf(
    x_plot,
    y_plot,
    z_plot,
    levels=70,
    cmap='jet'
)
ax.set_box_aspect(1)
fig.colorbar(
    cntr,
    ax=ax,
    shrink=0.74,
    pad=0.04
)
ax.set_title(r'Pointwise error $|u(x,t)-u_{\theta}(x,t)|$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
plt.tight_layout()
fig.savefig('Problem_D/pointwise_error_u.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



#material field a(x) (1D)
sample_idx = 0
a_show_1d = a_test[sample_idx]
x_show = x_mesh.ravel()
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x_show, a_show_1d, linewidth=1, color='C1')
ax.set_box_aspect(1)
ax.set_title(r'Initial velocity profile $a(x)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$a(x)$')
plt.tight_layout()
fig.savefig('Problem_D/input_a_1D.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()