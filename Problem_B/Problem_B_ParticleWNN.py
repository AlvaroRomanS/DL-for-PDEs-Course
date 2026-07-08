import numpy as np
import torch 
import torch.nn as nn
import h5py
device = 'cuda:0'
#


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


class MLP(nn.Module):

    def __init__(self, layers_list:list, dtype=None):
        super(MLP, self).__init__()
        # Network Sequential
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:]:
            net.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in = hidden
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # Input and hidden layers
        for net in self.net[:-1]:
            x = net(x)
            x = torch.sin(np.pi*x + np.pi) #custom activation function
            #x = torch.sin(np.pi*x+np.pi) + torch.sin(x)
        # Output layer
        x = self.net[-1](x)
        #x = torch.exp(-x**2)

        return x
#
'''
model_u = MLP([2, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
from torchsummary import summary
summary(model_u, input_size=(2,), device='cuda')
'''




lb, ub = np.array([0., 0.]), np.array([1., 1.])
#
nc = 10000
n_bd_each_side = 250
dataType = torch.float32
np.random.seed(1234)

'''
###### The truth solution: for generating the testing dataset, the initial condition, and the boundary condition
def u_true(x):
    # The ground truth solution
    x1, x2 = x[...,0:1], x[...,1:2]
    #
    u = torch.sin(torch.pi*x1)*torch.sin(torch.pi*x2)
    
    return u
'''
'''


# ################################# The boundary points
# The upper and lower boundaries
x_lw = np.vstack((np.linspace(lb[0], ub[0], n_bd_each_side), lb[0]*np.ones(n_bd_each_side))).T
x_up = np.vstack((np.linspace(lb[0], ub[0], n_bd_each_side), ub[0]*np.ones(n_bd_each_side))).T
# The left and right boundaries
x_lt = np.vstack((lb[1]*np.ones(n_bd_each_side), np.linspace(lb[1], ub[1], n_bd_each_side))).T
x_rt = np.vstack((ub[1]*np.ones(n_bd_each_side), np.linspace(lb[1], ub[1], n_bd_each_side))).T
# the boundary condition
x_bd = np.concatenate([x_lw, x_up, x_lt, x_rt], axis=0)
x_bd = torch.tensor(x_bd, dtype=dataType)
u_bd = torch.zeros_like(x_bd[:,0:1])
'''

################################# The particles (their corresponding radii are set as $R=0.001$）
xc = np.random.uniform(lb, ub, (nc,2))
xc = torch.tensor(xc, dtype=dataType)

# ################################# The boundary points
# The upper and lower boundaries
x_lw = np.vstack((np.linspace(lb[0], ub[0], n_bd_each_side), lb[0]*np.ones(n_bd_each_side))).T
x_up = np.vstack((np.linspace(lb[0], ub[0], n_bd_each_side), ub[0]*np.ones(n_bd_each_side))).T
# The left and right boundaries
x_lt = np.vstack((lb[1]*np.ones(n_bd_each_side), np.linspace(lb[1], ub[1], n_bd_each_side))).T
x_rt = np.vstack((ub[1]*np.ones(n_bd_each_side), np.linspace(lb[1], ub[1], n_bd_each_side))).T
# Boundary condition values (one scalar per side)
u_lw_vals = np.linspace(1.0, 0.0, n_bd_each_side).reshape(-1, 1)
u_up_vals = np.linspace(1.0, 0.0, n_bd_each_side).reshape(-1, 1)
u_lt_vals = np.ones((n_bd_each_side, 1))
u_rt_vals = np.zeros((n_bd_each_side, 1))

x_bd = np.concatenate([x_lw, x_up, x_lt, x_rt], axis=0)
x_bd = torch.tensor(x_bd, dtype=dataType)
u_bd = torch.tensor(
    np.concatenate([u_lw_vals, u_up_vals, u_lt_vals, u_rt_vals], axis=0),
    dtype=dataType
)




'''
# ################################# The testing dataset
xx, yy = np.meshgrid(np.linspace(lb[0], ub[0], 100), np.linspace(lb[1], ub[1], 100))
x_test = np.vstack([xx.flatten(), yy.flatten()]).T
# testing dataset
x_test = torch.tensor(x_test, dtype=dataType)
u_test = u_true(x_test)

# ##################################
'''
# DATA IMPORT
with h5py.File('Problem_B/ProblemB_dataset.h5', 'r') as f:
    print('The dataset for Problem B:', f.keys())
    mu_field = torch.tensor(np.array(f['mu_field']), dtype=torch.float32)
    x_test = torch.tensor(np.array(f['x_test']), dtype=torch.float32)
    u_test = torch.tensor(np.array(f['u_test']), dtype=torch.float32)

print('mu_field shape:', mu_field.shape)
print('x_test shape:', x_test.shape)
print('u_test shape:', u_test.shape)

mu_np = mu_field.numpy()
u_np = u_test.numpy()
x_np = x_test.numpy()

extent = [x_np[:, 0].min(), x_np[:, 0].max(), x_np[:, 1].min(), x_np[:, 1].max()]



from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    'gray_green',
    ["#E0E0E0", "#008A00"]
) #  plot of permeability field

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    mu_np,
    origin='lower',
    extent=extent,
    cmap=cmap,
    aspect='equal',
    interpolation='nearest'
)
fig.colorbar(im, ax=ax, label=r'$\mu(x)$', shrink=0.74)
ax.set_title(r'Permeability field $\mu(x)$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_B/mu_field.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)


'''
plt.subplot(1, 3, 2)
plt.scatter(x_np[:, 0], x_np[:, 1], s=0.5, c='steelblue', alpha=0.4)
plt.title('Test points $x_{\\mathrm{test}}$')
plt.xlabel('x'); plt.ylabel('y')
plt.gca().set_aspect('equal')

plt.subplot(1, 3, 3)
plt.imshow(u_np, origin='lower', extent=extent, cmap='RdBu_r', aspect='equal')
plt.colorbar(label='u')
plt.title('Reference (true) pressure field u(x)')
plt.xlabel('x'); plt.ylabel('y')


plt.suptitle('Problem B — Dataset overview', fontsize=13)
plt.tight_layout()
plt.savefig('problem_B_overview.png', dpi=150, bbox_inches='tight')
#plt.show()
plt.close()
'''



fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].scatter(x_test[...,0], x_test[...,1], label='Testing points')
axes[0].scatter(x_bd[:,0], x_bd[:,1], label='Boundary points')
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
axes[0].legend()
#
axes[1].scatter(xc[:,0], xc[:,1], color='g', label='Particles')
axes[1].set_xlabel('t')
axes[1].set_ylabel('x')
axes[1].legend()
#plt.show()
plt.close()






import math 

class TestFun():

    def __init__(self, dim:int):
        self.dim = dim

    def _dist(self, x:torch.tensor)->torch.tensor:
        '''
        Input:
          x: (?,d) or (?, m, d)
        Output:
          y: the norm of x
        '''
        return torch.linalg.norm(x, dim=-1, keepdims=True)

    def Wendland(self, grids)->torch.tensor:
        '''
        Input:
            grids: size(?,1), the number of meshgrids
        Output:
            v: the test function values
            dv: the grad dv/dx_mesh
        '''
        ############ Compute v and dv
        l = math.floor(self.dim / 2) + 3
        #
        r = 1. - torch.relu(1. - self._dist(grids))
        r_list = [r]
        for _ in range(1):
            r_list.append(r*r_list[-1])
        #
        v = (1-r) ** (l+2) * ( (l**2+4.*l+3.) * r_list[1] + (3.*l+6.) * r + 3.) / 3.
        #
        dv_dr_divide_by_r = (1-r)**(l+1) * (- (l**3+8.*l**2+19.*l+12) * r - (l**2+7.*l+12)) / 3.
        if self.dim==1: # 1D case
            dv = dv_dr_divide_by_r * r * torch.sign(grids)
        else: # 2D case
            dv =  dv_dr_divide_by_r * grids
            
        return v.detach(), dv.detach()
    


##### Generate intergal points in the ball B(xc, R)
n_mesh = 100
x_mesh, y_mesh = np.meshgrid(np.linspace(-1., 1., n_mesh), np.linspace(-1., 1., n_mesh))
grids = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
index = np.where(np.linalg.norm(grids, axis=1, keepdims=True) <1.)[0]
grids = torch.tensor(grids[index,:])

##### Get the test functions
GetTest = TestFun(dim=2)
v, dv = GetTest.Wendland(grids)
print(v.shape, dv.shape)

#### Visualize the test function (2d Wendland's CSRBF)
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#
g = grids.numpy()
x, y = g[:, 0], g[:, 1]
v_np = v.numpy().squeeze()
dvx = dv.numpy()[:, 0]
dvy = dv.numpy()[:, 1]
grad_mag = np.sqrt(dvx**2 + dvy**2)
 
triang = tri.Triangulation(x, y)
theta = np.linspace(0, 2 * np.pi, 300)
 
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
fig.suptitle('Wendland Test Function on the standard ball B(0,1)', fontsize=14, y=0.96)
 
panels = [
    (axes[0, 0], v_np,     'v(x,y)',    'Blues'),
    (axes[0, 1], dvx,      '∂v/∂x',     'RdBu_r'),
    (axes[1, 0], dvy,      '∂v/∂y',     'RdBu_r'),
    (axes[1, 1], grad_mag, '|∇v|',      'YlOrRd'),
]
 
for ax, field, title, cmap in panels:
    tc = ax.tricontourf(triang, field, levels=50, cmap=cmap)
    ax.tricontour(triang, field, levels=8, colors='white', linewidths=0.4, alpha=0.45)
    fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.35)
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
 
plt.tight_layout()
#plt.show()
plt.close()




import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad, Variable
device = 'cuda:0'
torch.manual_seed(1234)
#
class LossClass(object):

    def __init__(self, u_model):
        ''' 
        Input:
            u_model: NN for approximating u
            dim: the problem dimension
        '''
        self.device = device
        self.u_model = u_model
        self.getLoss = torch.nn.MSELoss()
        self.GetTest = TestFun(dim=2)

        self.mu_field = mu_field          # (128, 128) tensor
        self.x_min = x_test[:, 0].min()  # domain bounds for normalisation
        self.x_max = x_test[:, 0].max()
        self.y_min = x_test[:, 1].min()
        self.y_max = x_test[:, 1].max()

    def get_test(self, xc, R: float, n_mesh: int):
        '''
        Input:
            xc:    support region centers
            R:     radius of support regions
            N_int: number of Gauss-Legendre points per dimension
        '''


        #1D Gauss-Legendre nodes and weights
        x_int, w_int = np.polynomial.legendre.leggauss(n_mesh)
        x_int = torch.tensor(x_int, dtype=dataType)
        w_int = torch.tensor(w_int, dtype=dataType)
        # 2D tensor product quadrature grid
        xi, xj = torch.meshgrid(x_int, x_int, indexing='ij')
        wi, wj = torch.meshgrid(w_int, w_int, indexing='ij')
        x_int_2d = torch.stack([xi.flatten(), xj.flatten()], dim=-1)
        w_int_2d = (wi * wj).flatten().unsqueeze(-1)



        # Test functions and their gradients on the reference element [-1,1]^2
        # v:  (N_int^2, 1),  dv: (N_int^2, 2)
        v, dv = self.GetTest.Wendland(x_int_2d)
        dv = dv / R
        x_int_2d = x_int_2d.unsqueeze(0)# Physical quadrature points
        xc = xc.unsqueeze(1)
        x = xc + x_int_2d * R

        return x, w_int_2d, v, dv
    
    def loss_bd(self, x_bd, u_bd):
        '''loss term related to boundary condition'''
        ############## Loss term related to the boundary condition
        x = x_bd.to(self.device)
        u = self.u_model(x)
        loss = self.getLoss(u, u_bd.to(self.device))
        
        return loss 

    def loss_pde(self, x, w, v, dv):
        '''
        Input:
            x:  (Nc, N_int^2, 2)
            w:  (N_int^2, 1)
            v:  (N_int^2, 1)
            dv: (N_int^2, 2)
        '''
        Nc = x.shape[0]
        N_q = x.shape[1]

        #u and grad(u) at quadrature points
        x_flat = Variable(x.reshape(-1, 2), requires_grad=True).to(self.device)
        u = self.u_model(x_flat)
        du = grad(inputs=x_flat, outputs=u,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True)[0]

        # Interpolate mu(x) at quadrature points via grid_sample
        mu_grid = self.mu_field.to(self.device).unsqueeze(0).unsqueeze(0)
        xy_norm = (2.0 * x_flat - 1.0).unsqueeze(0).unsqueeze(0)
        mux = torch.nn.functional.grid_sample(
                    mu_grid, xy_norm, mode='bilinear', align_corners=True
                ).squeeze().reshape(-1, 1)


        du = du.reshape(Nc, N_q, 2)
        mux = mux.reshape(Nc, N_q, 1)
        dv = dv.to(self.device)
        w = w.to(self.device)

        # WEAK FORM RESIDUAL
        integrand = torch.sum(mux * du * dv.unsqueeze(0), dim=-1, keepdim=True)
        left = torch.sum(integrand * w.unsqueeze(0) * 0.25, dim=1)
        # 0.25 = (R/1)^2 * (1/2)^2 accounts for change of variables from [-1,1]^2
        right = 0.0
        loss = torch.mean((left - right) ** 2)

        return loss

    def get_error(self, x_test, u_test):
        '''Compute the L^2 relative error when testing dataset is given'''
        x_test = x_test.to(self.device)
        u_test = u_test.to(self.device)
        u = self.u_model(x_test)
        u = self.u_model(x_test).reshape(u_test.shape)
        
        return torch.sqrt(torch.sum((u-u_test)**2)/torch.sum(u_test**2))

################################### The bump function
'''
lossClass = LossClass(model_u)
x, w, v, dv = lossClass.get_test(xc, R=0.001, n_mesh=10)
print('x:', x.shape, 'v', v.shape, 'dv', dv.shape)
loss_pde = lossClass.loss_pde(x, w, v, dv)
'''





############ Training setups
from tqdm import trange
import time
from torch.utils.data import Dataset, DataLoader
#
class MyDataset(Dataset):

    def __init__(self, x):
        self.x = x.reshape(-1, 2)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]
        
dataloader = DataLoader(MyDataset(xc), batch_size=nc//50, shuffle=True)

#
epochs = 60 #best performance so far with 60
model_u = MLP([2, 200, 50, 50, 100, 1], dtype=torch.float32).to(device)
optimizer = torch.optim.Adam(params=model_u.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

from torchsummary import summary
summary(model_u, input_size=(2,), device='cuda')

#
w_pde = 1.
w_bd = 1000000.
############ The training process>
error_list, u_list, t_list = [], [], []
lossClass = LossClass(model_u)
t0 = time.time()
for epoch in trange(epochs):
    for xc_batch in dataloader:
        loss_bd = lossClass.loss_bd(x_bd, u_bd)
        x, w, v, dv = lossClass.get_test(xc_batch, R=0.001, n_mesh=15)
        loss_pde = lossClass.loss_pde(x, w, v, dv)
        loss_train = w_pde*loss_pde + w_bd*loss_bd
        #
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    t_list.append(time.time()-t0)
    # -------- Evaluation --------
    with torch.no_grad():
        error = lossClass.get_error(x_test.to(device), u_test.to(device))
        error_list.append(error.item())
        #
        u_pred = model_u(x_test.to(device)).detach().cpu()
        u_list.append(u_pred)
    #
    scheduler.step()  # Adjust learning rate
    if (epoch+1)%5==0:
        print(f'Epoch:{epoch}, The loss is:{loss_train.item()}, lr: {scheduler.optimizer.param_groups[0]["lr"]}')
        print(error.item())
    print(f'\nWeighted Boundary loss = {w_bd*loss_bd}')
    print(f'Weighted PDE loss = {w_pde*loss_pde}\n')





from scipy.interpolate import griddata

####### Make prediction with the trained model
x1_grid = np.linspace(lb[0], ub[0], 100)
x2_grid = np.linspace(lb[1], ub[1], 100)
x_mesh = np.meshgrid(x1_grid, x2_grid)
x_query = np.vstack((x_mesh[0].flatten(), x_mesh[1].flatten())).T
x_query = torch.tensor(x_query, dtype=dataType)

u_query = model_u(x_query.to(device)).detach().cpu()

####### Interpolate onto plot grid
x1_plot, x2_plot = np.meshgrid(np.linspace(lb[0], ub[0], 100), np.linspace(lb[1], ub[1], 100))

x1_ref = np.linspace(lb[0], ub[0], 128)
x2_ref = np.linspace(lb[1], ub[1], 128)
x_ref_mesh = np.meshgrid(x1_ref, x2_ref)
x_ref_points = np.vstack((x_ref_mesh[0].flatten(), x_ref_mesh[1].flatten())).T

z_pred = griddata((x_query[:, 0].numpy(), x_query[:, 1].numpy()), np.ravel(u_query), (x1_plot, x2_plot), method='cubic')
z_ref = griddata(x_ref_points, u_test.numpy().flatten(),                              (x1_plot, x2_plot), method='cubic')
z_err = np.abs(z_ref - z_pred)

####### Visualize the solution
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

vmin = min(z_pred.min(), z_ref.min())
vmax = max(z_pred.max(), z_ref.max())
err_vmin, err_vmax = 0, z_err.max()

for ax, z, title, cmin, cmax in zip(
    axs,
    [z_pred, z_ref, z_err],
    [r'Prediction $u_{\theta}(x)$', 'Ground truth $u(x)$', 'Point-wise absolute error'],
    [vmin, vmin, err_vmin],
    [vmax, vmax, err_vmax]
):
    cntr = ax.contourf(x1_plot, x2_plot, z, levels=70, cmap='jet', vmin=cmin, vmax=cmax)
    fig.colorbar(cntr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
plt.close()


# u prediction
fig, ax = plt.subplots(figsize=(5, 5))
cntr = ax.contourf(
    x1_plot, x2_plot, z_pred,
    levels=70, cmap='jet',
    vmin=vmin, vmax=vmax
)
ax.set_box_aspect(1)
# Colorbar with the same height as the square plot
cbar = fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
ax.set_title(r'Prediction $u_{\theta}(x)$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.tight_layout()
fig.savefig('Problem_B/predicted_u.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



# u GT
fig, ax = plt.subplots(figsize=(5, 5))
cntr = ax.contourf(
    x1_plot, x2_plot, z_ref,
    levels=70, cmap='jet',
    vmin=vmin, vmax=vmax
)
ax.set_box_aspect(1)
# Colorbar with the same height as the square plot
cbar = fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
ax.set_title(r'Ground truth $u(x)$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.tight_layout()
fig.savefig('Problem_B/GT_u.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



# u pointwise error
fig, ax = plt.subplots(figsize=(5, 5))
cntr = ax.contourf(
    x1_plot, x2_plot, z_err,
    levels=70, cmap='jet'
)
ax.set_box_aspect(1)
# Colorbar with the same height as the square plot
cbar = fig.colorbar(cntr, ax=ax, shrink=0.74, pad=0.04)
ax.set_title(r'Pointwise error $|u_{\theta}(x) - u(x)|$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.tight_layout()
fig.savefig('Problem_B/pointwise_error_u.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()














error = torch.sqrt(torch.sum((torch.tensor(z_ref) - torch.tensor(z_pred))**2) / torch.sum(torch.tensor(z_ref)**2))


# Error vs epoch
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(error_list, label='$L^2$ rel. error')
ax.set_title(f'Final $L^2$ error: $u = {error:.4f}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('$L^2$ relative error')
ax.legend()
#ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_B/error_vs_epoch.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()




'''
plt.figure(figsize=(7, 5))
plt.semilogy(t_list, error_list, label='$L^2$ relative error')
plt.title(f'$L^2$ relative error: {error:.4f}')
plt.xlabel('time (s)')
plt.ylabel('error')
plt.legend()
plt.tight_layout()
fig.savefig('Problem_B/error_vs_epoch.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

'''

print('END')