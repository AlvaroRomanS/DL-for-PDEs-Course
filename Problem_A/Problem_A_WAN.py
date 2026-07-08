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




fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].plot(x_test, u_test, color='k', label='True u(x)')
axes[0].scatter(x_obs, u_obs, color='r', label='Noisy observations')
axes[0].legend()
#
axes[1].plot(x_test, k_test, label='True k(x)')
axes[1].legend()
plt.tight_layout()
#plt.show()
plt.close()





#############################################################
# Step 1: Approximating the Solutions with Neural Networks
#############################################################
import torch.nn as nn
from torch.autograd import grad, Variable
torch.manual_seed(1234)

class MLP_uv(nn.Module):

    def __init__(self, layers_list:list, dtype=None):
        super(MLP_uv, self).__init__()
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
            x = torch.sin(np.pi*x+np.pi) + torch.sin(x)
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
            x = torch.sin(np.pi*x+np.pi) + torch.sin(x)
        # Output layer
        x = self.net[-1](x)

        return nn.functional.softplus(x)
#
model_u = MLP_uv([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
model_v = MLP_uv([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
model_k = MLP_k([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
from torchsummary import summary
summary(model_k, input_size=(1,), device='cuda')

#############################################################
# Step 2: Obtain the integral points and define test functions
#############################################################
L = 1.0
lb, ub = [0.], [L]
n_int = 100
################################# The boundary points (x=0) and (x=1)
x_lb = np.array(lb)
x_rb = np.array(ub)
#
x_bd = np.vstack([x_lb, x_rb])
x_bd = torch.tensor(x_bd, dtype=dataType)
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
class LossClass(object):

    def __init__(self, u_model, v_model, k_model):
        ''' 
        Input:
            u_model: NN for approximating u
        '''
        self.device = device
        self.u_model = u_model
        self.v_model = v_model
        self.k_model = k_model
        self.getLoss = torch.nn.MSELoss()
        self.scale = 0.110987

    def fun_bump(self, x):
        '''the bump function'''
        value = torch.where(torch.abs(x)<1., torch.exp(-1.0 / ((1+x) * (1 - x)))/self.scale, torch.zeros_like(x))
        return value

    def grad_bump(self, x):
        ''' '''
        value = torch.where(torch.abs(x)<1., torch.exp(-1.0 / ((1+x) * (1 - x))) * (-2*x)/((1+x)*(1-x))**2/self.scale, 
                            torch.zeros_like(x))
        return value
    
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
        
    def loss_pde(self, x, w):
        '''loss term related to the PDE'''
        ################# The PDE loss
        x = Variable(x.reshape(-1,1), requires_grad=True).to(self.device)
        w = w.reshape(-1,1).to(x)
        bump, dbump = self.fun_bump(x), self.grad_bump(x)
        #
        u = self.u_model(x)
        v = self.v_model(x)
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dv = grad(inputs=x, outputs=v, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        #
        kx = self.k_model(x)
        #a = a_freq.to(x)
        #fx = (1+torch.sin(a*x)**2)*(kk*np.pi)**2*torch.sin(kk*np.pi*x) - 2*a*kk*np.pi*torch.sin(a*x)*torch.cos(a*x)*torch.cos(kk*np.pi*x)
        fx = 9.81 # constant right hand side
        # The weak form
        residual_bump = (torch.sum(kx*du*dbump*w) - torch.sum(fx*bump*w))**2
        norm_bump = torch.sum(bump**2)
        loss_int_bump = residual_bump/norm_bump
        #
        residual = (torch.sum(kx*du*(dv*bump+v*dbump)*w) - torch.sum(fx*v*bump*w))**2
        norm_v = torch.mean((v*bump)**2)
        loss_int = residual/norm_v
        
        return loss_int, loss_int_bump

    def get_error(self, x_test, u_test, k_test):
        '''Compute the L^2 relative error when testing dataset is given'''
        x_test = x_test.to(self.device)
        u_test = u_test.to(self.device)
        k_test = k_test.to(self.device)
        k = self.k_model(x_test)
        u = self.u_model(x_test)
        
        return torch.sqrt(torch.sum((u-u_test)**2)/torch.sum(u_test**2)), torch.sqrt(torch.sum((k-k_test)**2)/torch.sum(k_test**2))

################################### The bump function
lossClass = LossClass(model_u, model_v, model_k)
x = torch.linspace(-1., 1., 100)
bump = lossClass.fun_bump(x)
dbump = lossClass.grad_bump(x)
#
plt.figure(figsize=(10,5))
plt.plot(x, bump, label='bump function')
plt.plot(x, dbump, label='grad_bump')
plt.title('The bump function')
plt.legend()
#plt.show()
plt.close()

#############################################################
# Step 4: Train the models
#############################################################
from tqdm import trange
import time
#
epochs = 500
lr_u, lr_v = 1e-3, 0.015
model_u = MLP_uv([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
model_v = MLP_uv([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
model_k = MLP_k([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
optimizer_u = torch.optim.Adam(params=list(model_u.parameters())+list(model_k.parameters()), lr=lr_u)
optimizer_v = torch.optim.Adagrad(params=model_v.parameters(), lr=lr_v)
scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=np.int32(epochs/10), gamma=2/3)
scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=np.int32(epochs/10), gamma=0.9)
#
w_bd = 1.
w_pde1 = 50.
w_pde2 = 50.
w_ob = 100.
steps_u, steps_v = 1, 1
############ The training process
error_u_list, error_k_list, u_list, v_list, t_list = [], [], [], [], []
lossClass = LossClass(model_u, model_v, model_k)
for epoch in trange(epochs):
    t0 = time.time()
    # -------- Update model_u and model_k（fixed model_v）--------
    for param in model_v.parameters():
        param.requires_grad_(False)
    for param in model_u.parameters():
        param.requires_grad_(True)
    for param in model_k.parameters():
        param.requires_grad_(True)
    for _ in range(steps_u):
        loss_bd = lossClass.loss_bd(x_bd, u_bd)
        loss_ob = lossClass.loss_obs(x_obs, u_obs)
        loss_int, loss_int_bump = lossClass.loss_pde(x=x_int, w=w_int)
        loss_train_u = w_pde1*loss_int + w_pde2*loss_int_bump + w_bd*loss_bd + w_ob * loss_ob
        #
        optimizer_u.zero_grad()
        loss_train_u.backward()
        optimizer_u.step()

    # -------- Update model_v（fixed model_u and model_k）--------
    for param in model_v.parameters():
        param.requires_grad_(True)
    for param in model_u.parameters():
        param.requires_grad_(False)
    for param in model_k.parameters():
        param.requires_grad_(False)
    for _ in range(steps_v):
        loss_int, _ = lossClass.loss_pde(x=x_int, w=w_int)
        loss_train_v = - torch.log(10*w_pde1*loss_int)
        #
        optimizer_v.zero_grad()
        loss_train_v.backward()
        optimizer_v.step()
    #
    t_list.append(time.time()-t0)
    
    # -------- Evaluation --------
    with torch.no_grad():
        error_u, error_k = lossClass.get_error(x_test.to(device), u_test.to(device), k_test.to(device))
        error_u_list.append(error_u.item())
        error_k_list.append(error_k.item())
        #
        u_pred = model_u(x_test.to(device)).detach().cpu()
        u_list.append(u_pred)
        v_pred = model_v(x_test.to(device)).detach().cpu() * lossClass.fun_bump(x_test)
        v_list.append(v_pred)
    #
    scheduler_u.step()  # Adjust learning rate
    scheduler_v.step()  # Adjust learning rate
    if (epoch+1)%100==0:
        print(f'Epoch:{epoch}, The loss_u is:{loss_train_u.item()}, loss_v is:{loss_train_v.item()}, lr_u: {scheduler_u.optimizer.param_groups[0]["lr"]}')
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
plt.close()