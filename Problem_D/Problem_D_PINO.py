import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import time
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

print('START')

setup_seed(3407)
device = 'cuda'
dtype = torch.float32




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


    a = torch.tensor(a_np, dtype=dtype).view(N, 1, N_x, 1).repeat(1, N_t, 1, 1)

    t_coord = torch.tensor(t_mesh, dtype=dtype).view(1, N_t, 1, 1).repeat(N, 1, N_x, 1)
    x_coord = torch.tensor(x_mesh, dtype=dtype).view(1, 1, N_x, 1).repeat(N, N_t, 1, 1)

    ax = torch.cat([a, t_coord, x_coord], dim=-1)

    u = None
    if u_np is not None:
        u = torch.tensor(u_np, dtype=dtype).unsqueeze(-1)

    return ax, u

ax_train_labeled, u_train_labeled = transform_data(a_train_labeled, x_mesh, t_mesh, u_train_labeled)
ax_train_unlabeled, _= transform_data(a_train_unlabeled, x_mesh, t_mesh, u_np=None)
ax_test, u_test = transform_data(a_test, x_mesh, t_mesh, u_test)

print('ax_train_labeled:', ax_train_labeled.shape)
print('u_train_labeled:', u_train_labeled.shape)
print('ax_train_unlabeled:', ax_train_unlabeled.shape)
print('ax_test:', ax_test.shape)
print('u_test:', u_test.shape)
print('gridx:', gridx.shape)


a_show = ax_train_unlabeled[0, ..., 0]

t_plot_lin = np.linspace(t_mesh.min(), t_mesh.max(), 100)
x_plot_lin = np.linspace(x_mesh.min(), x_mesh.max(), 200)
t_plot, x_plot = np.meshgrid(t_plot_lin, x_plot_lin, indexing='ij')

fig, ax = plt.subplots(figsize=(6, 4))
z_plot = griddata((gridx[:,0], gridx[:,1]), np.ravel(a_show), (t_plot, x_plot), method='cubic')
cntr = ax.contourf(t_plot, x_plot, z_plot, levels=40, cmap='jet')
fig.colorbar(cntr, ax=ax)
ax.set_title('Input initial condition a(x) broadcast over t (unlabeled)')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.show()





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
        batch_size = x.shape[0]
        n_t, n_x = x.size(-2), x.size(-1)


        n_t_pad = 2 ** math.ceil(math.log2(n_t))   # 200 -> 256
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, n_t_pad - n_t))

        x_ft = torch.fft.rfft2(x_pad)
        out_ft = torch.zeros(batch_size, self.out_size, n_t_pad, n_x // 2 + 1,
                            device=x.device, dtype=torch.cfloat)
        out_ft[:, :,  :self.modes1, :self.modes2] = \
            self.compl_mul_2d(x_ft[:, :,  :self.modes1, :self.modes2], self.weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul_2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weight2)

        out = torch.fft.irfft2(out_ft, s=(n_t_pad, n_x))
        return out[:, :, :n_t, :]


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
        self.spectral_conv = nn.ModuleList(conv_net)
        self.weight_conv = nn.ModuleList(w_net)
        # The output layer
        self.fc_out0 = nn.Linear(self.hidden_in, 128, dtype=dtype)
        self.fc_out1 = nn.Linear(128, out_size,   dtype=dtype)

    def forward(self, ax):
        '''
        Input: 
            ax: size(batch_size, my_size, mx_size, a_size+x_size)
        Output: 
            u: size(batch_size, my_size, mx_size, out_size)
        '''
        batch_size = ax.shape[0]
        mx_size, my_size = ax.shape[1], ax.shape[2]
        ax = self.fc_in(ax).permute(0, 3, 1, 2)
        hidden_last = self.hidden_list[0]
        for conv, weight, hidden_size in zip(self.spectral_conv, self.weight_conv, self.hidden_list[1:]):
            ax1 = conv(ax)
            ax2 = weight(ax.view(batch_size, hidden_last, -1)).view(batch_size, hidden_size, mx_size, my_size)
            ax = self.activation(ax1 + ax2)
            hidden_last = hidden_size
        ax = ax.permute(0, 2, 3, 1)
        ax = self.activation(self.fc_out0(ax))
        return self.fc_out1(ax)


mode1, mode2 = 12, 12
hidden_list = [40, 40, 40]
model_u = FNO2d(ax_train_labeled.shape[-1],
                u_train_labeled.shape[-1],
                mode1, mode2, hidden_list).to(device)

total_params = sum(p.numel() for p in model_u.parameters() if p.requires_grad)
print(f'{total_params:,} trainable parameters.')





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


class mollifier(object):

    def __call__(self, u_tilde, ax):
        '''
        Input:
            u_tilde : (B, N_t, N_x, 1)  — raw FNO output
            ax      : (B, N_t, N_x, 3)  — channels: a(x), t, x
        Output:
            u       : (B, N_t, N_x, 1)  — physically constrained field
        '''
        a = ax[..., 0:1]   # (B, N_t, N_x, 1) — initial condition, broadcast across t
        t = ax[..., 1:2]   # (B, N_t, N_x, 1)
        x = ax[..., 2:3]   # (B, N_t, N_x, 1)

        spatial_mask = torch.sin(np.pi * (x + 1) / 2)   # vanishes at x = ±1
        u = a + t * spatial_mask * u_tilde               # IC satisfied at t=0, BCs at x=±1
        return u

######################################
# Loss
######################################
nu = 0.1 / np.pi

class LossClass(object):

    def __init__(self, u_model, lambda_r=1.0, lambda_d=1.0):
        """
        Args:
            u_model  : FNO2d
            lambda_r : weight for PDE residual loss
            lambda_d : weight for supervised data loss (0 = fully physics-informed)
        """
        self.u_model = u_model
        self.mollifier = mollifier()
        self.lambda_r = lambda_r
        self.lambda_d = lambda_d

        self.dt = float(t_mesh[1] - t_mesh[0])   # uniform time spacing
        self.dx = float(x_mesh[1] - x_mesh[0])   # uniform space spacing

    def _predict(self, ax_batch):
        """Forward pass with normalisation and ansatz. Returns u: (B, N_t, N_x)."""
        ax_norm = normalizer_ax.encode(ax_batch)
        u_tilde = self.u_model(ax_norm)                  # (B, N_t, N_x, 1)
        u = self.mollifier(u_tilde, ax_batch)      # (B, N_t, N_x, 1)
        return u[..., 0]                                 # (B, N_t, N_x)

    def loss_pde(self, ax_batch):
        """
        Burgers' residual: du/dt + u*du/dx - nu*d²u/dx² = 0
        Central differences in both t and x on interior points.
        Residual computed on interior: t in [1:-1], x in [1:-1]  ->  (B, N_t-2, N_x-2)
        """
        n_batch = ax_batch.shape[0]
        u = self._predict(ax_batch)                      # (B, N_t, N_x)

        # du/dt — central difference along axis 0 (time)
        dudt = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * self.dt)   # (B, N_t-2, N_x-2)

        # du/dx — central difference along axis 1 (space)
        dudx = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * self.dx)   # (B, N_t-2, N_x-2)

        # d²u/dx² — second-order central difference along axis 1 (space)
        d2udx2 = (u[:, 1:-1, 2:] - 2*u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (self.dx**2)  # (B, N_t-2, N_x-2)

        # Burgers' residual
        u_interior = u[:, 1:-1, 1:-1]                                 # (B, N_t-2, N_x-2)
        residual = dudt + u_interior * dudx - nu * d2udx2           # (B, N_t-2, N_x-2)

        loss = torch.norm(residual.reshape(n_batch, -1), 2, dim=1)
        return torch.mean(loss)

    def loss_data(self, ax_batch, u_batch):
        """
        L_data = mean_i || u_pred_i - u_true_i ||_2
        Only called on labeled pairs.
        """
        batch_size = u_batch.shape[0]
        u_pred = self._predict(ax_batch)             # (B, N_t, N_x)
        u_true = u_batch[..., 0]                     # (B, N_t, N_x)
        loss = torch.norm(
            (u_true - u_pred).reshape(batch_size, -1), 2, dim=1
        )
        return torch.mean(loss)

    def loss_total(self, ax_unlabeled, ax_labeled=None, u_labeled=None):
        # Single PDE forward pass over all samples at once
        if ax_labeled is not None:
            ax_all = torch.cat([ax_unlabeled, ax_labeled], dim=0)
        else:
            ax_all = ax_unlabeled

        loss = self.lambda_r * self.loss_pde(ax_all)

        if self.lambda_d > 0 and ax_labeled is not None:
            loss = loss + self.lambda_d * self.loss_data(ax_labeled, u_labeled)
        return loss

    def get_error(self, ax, u):
        """L2 relative error on labeled pairs."""
        batch_size = u.shape[0]
        u_pred = self._predict(ax)                       # (B, N_t, N_x)
        u_true = u[..., 0]                               # (B, N_t, N_x)
        error = (torch.norm((u_true - u_pred).reshape(batch_size, -1), 2, dim=1)
                / torch.norm( u_true          .reshape(batch_size, -1), 2, dim=1))
        return torch.mean(error)

loss_fn = LossClass(model_u, lambda_r=1.0, lambda_d=1.0)
    



######################################
# Dataset — unchanged
######################################
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




######################################
# Training
#
# PINO mode        (lambda_d = 0): PDE loss on all data, no labeled u.
# Semi-supervised  (lambda_d > 0): PDE loss on all data + data loss on labeled pairs.
######################################
LAMBDA_R = 1.0
LAMBDA_D = 1.0

BATCH_SIZE_UNLABELED = 40
BATCH_SIZE_LABELED = 10    # smaller — only 200 labeled samples vs 1800 unlabeled

# Unlabeled loader: ax only, used for PDE loss
unlabeled_dataset = MyDataset(ax_train_unlabeled)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE_UNLABELED, shuffle=True)

# Labeled loader: ax + u, used for both PDE loss and data loss
labeled_dataset = MyDataset(ax_train_labeled, u_train_labeled)
labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE_LABELED, shuffle=True)

loss_fn = LossClass(model_u, lambda_r=LAMBDA_R, lambda_d=LAMBDA_D)

epochs = 200
lr = 1e-3
optimizer = torch.optim.Adam(model_u.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=np.int32(epochs/5), gamma=0.5)

loss_list, error_list = [], []
t0 = time.time()


def evaluate_error(loss_fn, ax, u, batch_size=5):
    errors = []
    for i in range(0, ax.shape[0], batch_size):
        ax_b = ax[i:i+batch_size].to(device)
        u_b = u[i:i+batch_size].to(device)
        with torch.no_grad():
            errors.append(loss_fn.get_error(ax_b, u_b).item())
    return np.mean(errors)



torch.cuda.empty_cache()
import gc; gc.collect()

print(torch.cuda.memory_allocated() / 1e9, 'GB allocated')
print(torch.cuda.memory_reserved()  / 1e9, 'GB reserved')
print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB total')


for epoch in trange(epochs):
    model_u.train()
    epoch_loss = 0.0

    # Zip the two loaders together. The labeled loader is shorter (200 vs 1800 samples),
    # so we cycle it so it keeps providing batches for every unlabeled batch.
    from itertools import cycle
    for unlabeled_batch, labeled_batch in zip(unlabeled_loader, cycle(labeled_loader)):
        ax_unlabeled = unlabeled_batch[0].to(device)
        ax_labeled = labeled_batch[0].to(device)
        u_labeled = labeled_batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)   # only here, before the forward pass

        t0 = time.time()
        loss_train = loss_fn.loss_total(ax_unlabeled, ax_labeled=ax_labeled, u_labeled=u_labeled)
        torch.cuda.synchronize()
        print(f'forward:  {time.time()-t0:.2f}s')

        t0 = time.time()
        loss_train.backward()
        torch.cuda.synchronize()
        print(f'backward: {time.time()-t0:.2f}s')

        t0 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        print(f'step:     {time.time()-t0:.2f}s')

        epoch_loss += loss_train.item()
        

    scheduler.step()

    # Evaluate on test set (monitoring only — never used for training)
    t0 = time.time()
    model_u.eval()
    with torch.no_grad():
        error = evaluate_error(loss_fn, ax_test, u_test, batch_size=5)
    torch.cuda.synchronize()
    print(f'evaluation: {time.time()-t0:.2f}s')
    error_list.append(error)

    epoch_loss /= len(unlabeled_loader)
    loss_list.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:4d} | Total loss: {epoch_loss:.4e} '
              f'| Test L2 rel. error: {error_list[-1]:.4e}')

print(f'\nTotal training time: {time.time() - t0:.1f}s')




#######################################
# L2 relative error — unchanged
#######################################
def L2_error(u, u_pred):
    l2 = torch.norm(u - u_pred, 2, 1) / torch.norm(u, 2, 1)
    return l2

######################################
# Generate predictions on test set
######################################
model_u.eval()
u_pred_chunks = []
for i in range(0, ax_test.shape[0], 5):
    ax_b = ax_test[i:i+5].to(device)
    with torch.no_grad():
        ax_norm = normalizer_ax.encode(ax_b)
        u_tilde = model_u(ax_norm)
        u_pred_b = mollifier()(u_tilde, ax_b).cpu()
    u_pred_chunks.append(u_pred_b)
u_pred_all = torch.cat(u_pred_chunks, dim=0)   # [200, N_t, N_x, 1]

# L2 relative error — flatten over the spatio-temporal grid [N_t * N_x]
l2_err = L2_error(u_test.reshape(200, -1), u_pred_all.reshape(200, -1))
print(f'Average L2 relative error: {torch.mean(l2_err):.4e}')

######################################
# Visualise predictions vs ground truth
# Rows: 3 test samples. Columns: input a(x), true u(x,t), predicted u(x,t), pointwise error
######################################
indices = [0, 1, 2]

# Plotting grid in (t, x) space — matches axis convention [time, space]
t_plot_lin = np.linspace(t_mesh.min(), t_mesh.max(), 100)
x_plot_lin = np.linspace(x_mesh.min(), x_mesh.max(), 200)
t_plot, x_plot = np.meshgrid(t_plot_lin, x_plot_lin, indexing='ij')

fig, axs = plt.subplots(nrows=len(indices), ncols=4, figsize=(20, 4 * len(indices)))

for row, idx in enumerate(indices):
    a_show = ax_test[idx, 0, :, 0].numpy()          # [N_x] — IC, slice at t=0
    u_show = u_test[idx, ..., 0].numpy()             # [N_t, N_x]
    u_pred_show = u_pred_all[idx, ..., 0].numpy()         # [N_t, N_x]
    err_show = np.abs(u_show - u_pred_show)            # [N_t, N_x]

    # Input a(x): 1D plot since it's only defined at t=0
    axs[row, 0].plot(x_mesh.ravel(), a_show)
    axs[row, 0].set_title(f'Input a(x) — sample {idx}')
    axs[row, 0].set_xlabel('x')
    axs[row, 0].set_ylabel('a')
    axs[row, 0].set_xlim([x_mesh.min(), x_mesh.max()])

    # True u(x,t), predicted u(x,t), pointwise error: 2D contour plots
    for col, (vals, title) in enumerate(zip(
            [u_show, u_pred_show, err_show],
            [f'True u — sample {idx}',
             f'Predicted u — sample {idx}',
             f'Pointwise |error| — sample {idx}'])):
        z = griddata((gridx[:, 0], gridx[:, 1]), vals.ravel(), (t_plot, x_plot), method='cubic')
        cntr = axs[row, col+1].contourf(t_plot, x_plot, z, levels=40, cmap='jet')
        fig.colorbar(cntr, ax=axs[row, col+1])
        axs[row, col+1].set_title(title)
        axs[row, col+1].set_xlabel('t')
        axs[row, col+1].set_ylabel('x')

plt.tight_layout()
plt.show()

######################################
# Training curves
######################################
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].semilogy(loss_list);  axs[0].set_title('Total Training Loss');    axs[0].set_xlabel('Epoch')
axs[1].semilogy(error_list); axs[1].set_title('Test L2 Relative Error'); axs[1].set_xlabel('Epoch')
plt.tight_layout()
plt.show()
plt.close()



print('END')