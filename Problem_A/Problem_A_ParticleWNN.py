import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
np.random.seed(1234)
dataType = torch.float32
device = 'cuda:0'


print('START')


with h5py.File('Problem_A/ProblemA_dataset.h5', 'r') as f:
    print('The dataset for Problem A:', f.keys())
    x_obs = torch.tensor(np.array(f['x_obs']), dtype=torch.float32)
    u_obs = torch.tensor(np.array(f['u_obs']), dtype=torch.float32)
    x_test = torch.tensor(np.array(f['x_test']), dtype=torch.float32)
    k_test = torch.tensor(np.array(f['k_test']), dtype=torch.float32)
    u_test = torch.tensor(np.array(f['u_test']), dtype=torch.float32)

    if x_obs.dim() == 1: x_obs = x_obs.unsqueeze(1)
    if u_obs.dim() == 1: u_obs = u_obs.unsqueeze(1)
    if x_test.dim() == 1: x_test = x_test.unsqueeze(1)
    if k_test.dim() == 1: k_test = k_test.unsqueeze(1)
    if u_test.dim() == 1: u_test = u_test.unsqueeze(1)

print('x_obs shape:', x_obs.shape)
print('u_obs shape:', u_obs.shape)
print('x_test shape:', x_test.shape)
print('k_test shape:', k_test.shape)
print('u_test shape:', u_test.shape)


'''
# The truth solution
kk, a_freq = 1, np.random.uniform(1, np.pi, 1)
x_test = np.linspace(-1, 1, 1000).reshape(-1,1)
u_test = np.sin(kk * np.pi * x_test)
k_test = 1. + np.sin(a_freq * x_test)**2
x_test = torch.tensor(x_test, dtype=dataType)
u_test = torch.tensor(u_test, dtype=dataType)
k_test = torch.tensor(k_test, dtype=dataType)
a_freq = torch.tensor(a_freq, dtype=dataType)

# The observation
noise_std = 0.05
x_obs = np.random.uniform(-1,1,(100,1))
u_obs_clean = np.sin(kk * np.pi * x_obs)
u_obs = u_obs_clean + np.random.normal(0, noise_std, u_obs_clean.shape)  # Add Gaussian noise
x_obs = torch.tensor(x_obs, dtype=dataType)
u_obs = torch.tensor(u_obs, dtype=dataType)
'''

fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].plot(x_test, u_test, color='k', label='True u(x)')
axes[0].scatter(x_obs, u_obs, color='r', label='Noisy observations')
axes[0].legend()
#
axes[1].plot(x_test, k_test, label='True k(x)')
axes[1].legend()
plt.tight_layout()
plt.show()




#############################################################
# Step 1: Approximating the Solutions with Neural Networks
#############################################################
import torch.nn as nn
from torch.autograd import grad, Variable
torch.manual_seed(1234)

class MLP_u(nn.Module):

    def __init__(self, layers_list:list, dtype=None):
        super(MLP_u, self).__init__()
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
            x = torch.sin(np.pi*x+np.pi) + torch.sin(x) # Custom activation function
        # Output layer
        x = self.net[-1](x)

        return x
#
class MLP_k(nn.Module):

    def __init__(self, layers_list:list, dtype=None):
        super(MLP_k, self).__init__()
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
            x = torch.sin(np.pi*x+np.pi) + torch.sin(x) # Custom activation function
        # Output layer
        x = self.net[-1](x)

        return nn.functional.softplus(x)
#
model_u = MLP_u([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
model_k = MLP_k([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
from torchsummary import summary
summary(model_u, input_size=(1,), device='cuda')

#############################################################
# Step 2: Obtain the integral points and define test functions
#############################################################
L = 1.0 # length of domain
lb, ub = [0.], [L]
n_int = 100
################################# The boundary points (x=0) and (x=1)
x_lb = np.array(lb)
x_rb = np.array(ub)
#
x_bd = np.vstack([x_lb, x_rb])
x_bd = torch.tensor(x_bd, dtype=dataType)
# Here is where we enforce the values of the BCs, in this case both 0.
u_bd = torch.tensor([[0.], [0.]], dtype=dataType)
print('Size of boundary points:', x_bd.shape)
################################# The integral points (x)
x_int, w_int = np.polynomial.legendre.leggauss(n_int)
x_int = torch.tensor(x_int, dtype=dataType).reshape(-1,1)
w_int = torch.tensor(w_int, dtype=dataType).reshape(-1,1)
print('Size of integral points:', x_int.shape)

#############################################################
# Step 3: Define the loss function 
#############################################################
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
            
        return v.detach(), dv.detach(), grids
#
class LossClass(object):

    def __init__(self, u_model, k_model, f_const):
        ''' 
        Input:
            u_model: NN for approximating u
            dim: the problem dimension
        '''
        self.device = device
        self.u_model = u_model
        self.k_model = k_model
        self.f_const = f_const
        self.getLoss = torch.nn.MSELoss()
        self.GetTest = TestFun(dim=1)

    def get_test(self, Nc:int, R:float, N_int:int):
        '''
        Input:
            Nc: the number of test functions
            R: the radius of the support region
            N_int: the number of integral points
        '''
        # Obtain centers of support regions and the corresponding radii
        xc = np.random.uniform(lb, ub, Nc).reshape(-1,1)
        xc = torch.tensor(xc, dtype=dataType).unsqueeze(1)
        # Obtain integral points, test function, and gradient of test function
        x_int, w_int = np.polynomial.legendre.leggauss(N_int)
        x_int = torch.tensor(x_int.reshape(-1,1), dtype=dataType)
        w_int = torch.tensor(w_int.reshape(-1,1), dtype=dataType)
        v, dv, grids = self.GetTest.Wendland(x_int)
        # 
        x = xc + x_int * R
        dv = dv/R
        
        return x, w_int, v, dv
    
    def loss_bd(self, x_bd, u_bd):
        '''loss term related to boundary condition'''
        ############## Loss term related to the boundary condition
        x = x_bd.to(self.device)
        u = self.u_model(x)
        loss = self.getLoss(u, u_bd.to(self.device))
        
        return loss 

    def loss_obs(self, x_obs, u_obs):
        '''loss term related to the observation'''
        x = x_obs.to(self.device)
        u = self.u_model(x)
        loss = self.getLoss(u, u_obs.to(x))
        
        return loss
        
    def loss_pde(self, x, w, v, dv):
        '''loss term related to the PDE
        Input: 
            x: size(Nc, N_int, 1)
            w: size(N_int, 1) 
            v: size(N_int, 1)
            dv: size(N_int, 1)
        '''
        Nc, N_int = x.shape[0], x.shape[1]
        ################# The PDE loss
        x = Variable(x.reshape(-1,1), requires_grad=True).to(self.device)
        ##a = a_freq.to(x)
        u = self.u_model(x)
        kx = self.k_model(x)
        # kx = 1 + torch.sin(a*x)**2
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        x, u, du = x.reshape(Nc, N_int, 1), u.reshape(Nc, N_int, 1), du.reshape(Nc, N_int, 1)
        kx = kx.reshape(Nc, N_int, 1)
        #
        ##fx = (1+torch.sin(a*x)**2)*(kk*np.pi)**2*torch.sin(kk*np.pi*x) - 2*a*kk*np.pi*torch.sin(a*x)*torch.cos(a*x)*torch.cos(kk*np.pi*x)
        fx = self.f_const
        # The weak form
        left = torch.sum(kx * du * dv.to(x) * w.to(x), dim=1)
        right = torch.sum(fx * v.to(x) * w.to(x), dim=1)
        loss = torch.mean((left-right)**2)
        
        return loss

    def get_error(self, x_test, u_test, k_test):
        '''Compute the L^2 relative error when testing dataset is given'''
        x_test = x_test.to(self.device)
        u_test = u_test.to(self.device)
        k_test = k_test.to(self.device)
        k = self.k_model(x_test)
        u = self.u_model(x_test)
        
        return torch.sqrt(torch.sum((u-u_test)**2)/torch.sum(u_test**2)), torch.sqrt(torch.sum((k-k_test)**2)/torch.sum(k_test**2))

################################### The bump function
f_const = 9.81 # right hand side (constant)
lossClass = LossClass(model_u, model_k, f_const)
x, w, v, dv = lossClass.get_test(Nc=5, R=0.1, N_int=100)
#
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(x[0], v, label=r'$v_1$')
axes[0].plot(x[1], v, label=r'$v_2$')
axes[0].plot(x[2], v, label=r'$v_3$')
axes[0].plot(x[3], v, label=r'$v_4$')
axes[0].plot(x[4], v, label=r'$v_5$')
axes[0].set_title('The test functions')
axes[0].legend()
#
axes[1].plot(x[0], dv, label=r'$dv_1$')
axes[1].plot(x[1], dv, label=r'$dv_2$')
axes[1].plot(x[2], dv, label=r'$dv_3$')
axes[1].plot(x[3], dv, label=r'$dv_4$')
axes[1].plot(x[4], dv, label=r'$dv_5$')
axes[1].set_title('The 1st gradients of test functions')
axes[1].legend()
plt.show()

#############################################################
# Step 4: Train the models
#############################################################
from tqdm import trange
import time
#
epochs = 500
model_u = MLP_u([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
model_k = MLP_k([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
optimizer = torch.optim.Adam(params=list(model_u.parameters())+list(model_k.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=np.int32(epochs/10), gamma=2/3)
#

# Check if running on GPU of CPU
print(next(model_u.parameters()).device)
print(next(model_k.parameters()).device)
print('CUDA available:', torch.cuda.is_available())
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')



w_bd = 1.
w_pde = 1.
w_ob = 20.
############ The training process
error_u_list, error_k_list, u_list, t_list = [], [], [], []
lossClass = LossClass(model_u, model_k, f_const)
for epoch in trange(epochs):
    t0 = time.time()
    loss_bd = lossClass.loss_bd(x_bd, u_bd)
    loss_ob = lossClass.loss_obs(x_obs, u_obs)
    x, w, v, dv = lossClass.get_test(Nc=2, R=0.2, N_int=20)
    loss_pde = lossClass.loss_pde(x, w, v, dv)
    loss_train = w_pde*loss_pde + w_bd*loss_bd + w_ob*loss_ob
    #
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    t_list.append(time.time()-t0)
    
    # -------- Evaluation --------
    with torch.no_grad():
        error_u, error_k = lossClass.get_error(x_test.to(device), u_test.to(device), k_test.to(device))
        error_u_list.append(error_u.item())
        error_k_list.append(error_k.item())
        #
        u_pred = model_u(x_test.to(device)).detach().cpu()
        u_list.append(u_pred)
    #
    scheduler.step()  # Adjust learning rate
    if (epoch+1)%100==0:
        print(f'Epoch:{epoch}, The loss is:{loss_train.item()}, lr: {scheduler.optimizer.param_groups[0]["lr"]}')
        print(error_u.item(), error_k.item())




u_query = model_u(x_test.to(device)).detach().cpu()
k_query = model_k(x_test.to(device)).detach().cpu()
error_u = torch.sqrt(torch.sum((u_test-u_query)**2)/torch.sum(u_test**2))
error_k = torch.sqrt(torch.sum((k_test-k_query)**2)/torch.sum(k_test**2))
#
fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].plot(x_test, u_query, label='Pred. u')
axes[0].plot(x_test, u_test, label='True u')
axes[0].set_title(f'$L^2$ relative error for u is:{error_u}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u')
axes[0].legend()
#
axes[1].plot(x_test, k_query, label='Pred. k')
axes[1].plot(x_test, k_test, label='True k')
axes[1].set_title(f'$L^2$ relative error for k is:{error_k}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('k')
axes[1].legend()
#
axes[2].semilogy(np.cumsum(t_list), error_u_list, label='$L^2$ relative error (u)')
axes[2].semilogy(np.cumsum(t_list), error_k_list, label='$L^2$ relative error (k)')
axes[2].set_title('$L^2$ relative error vs. times')
axes[2].set_xlabel('time (s)')
axes[2].set_ylabel('error')
axes[2].legend()
#
plt.show()




print('END')