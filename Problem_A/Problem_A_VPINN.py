import numpy as np
import torch

import h5py
np.random.seed(1234)
dataType = torch.float32
device = 'cuda:0'

import matplotlib.font_manager as fm
import matplotlib as mpl

# find font library
matches = [f.name for f in fm.fontManager.ttflist if 'CMU' in f.name]
print(matches)  # should include 'CMU Sans Serif'
print(mpl.get_cachedir())

# set fonts for figures
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['CMU Sans Serif']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'CMU Sans Serif'
mpl.rcParams['mathtext.it'] = 'CMU Sans Serif:italic'
mpl.rcParams['mathtext.bf'] = 'CMU Sans Serif:bold'
#mpl.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt


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
axes[1].plot(x_test, k_test, label=r'True $k(x)$')
axes[1].legend()
plt.tight_layout()
plt.show()
plt.close()



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
            x = torch.sin(np.pi*x+np.pi) + torch.sin(x)
            #x = torch.cos(1*np.pi*x) + torch.cos(.5*np.pi*x)
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
            #x = torch.cos(1*np.pi*x) + torch.cos(.5*np.pi*x)
        # Output layer
        x = self.net[-1](x)

        return nn.functional.softplus(x)
#
##model_u = MLP_u([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
##model_k = MLP_k([1, 60, 60, 60, 60, 1], dtype=torch.float32).to(device)
##from torchsummary import summary
##summary(model_u, input_size=(1,), device='cuda')

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
from scipy.special import jacobi
# Recursive generation of the Jacobi polynomial of order n
def Jacobi(n,a,b,x):
    x=np.array(x)
    return (jacobi(n,a,b)(x))
#
class TestFun():

    def __init__(self, N_test:int):
        self.N_test = N_test

    def get_v(self, x):
        '''
        Input:
            x: size(n,); coordinates in [-1,1]
        Output:
            v: size(N_test, n); test functions
        '''
        v_total = []
        for n in range(1, self.N_test+1):
            v = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
            v_total.append(v)
        return np.asarray(v_total)

    def get_dv(self, x):
        '''
        Input:
            x:size(n,); coordinates in [-1,1]
        Output:
            dv: size(N_test, n); 1st gradient of test functions
            ddv: size(N_test, n); 2nd gradient of test functions
        '''
        dv_total = []
        ddv_total = []
        for n in range(1, self.N_test+1):
            if n==1:
                dv = ((n+2)/2)*Jacobi(n,1,1,x)
                ddv = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                dv_total.append(dv)
                ddv_total.append(ddv)
            elif n==2:
                dv = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                ddv = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                dv_total.append(dv)
                ddv_total.append(ddv)    
            else:
                dv = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                ddv = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
                dv_total.append(dv)
                ddv_total.append(ddv)    
                
        return np.asarray(dv_total), np.asarray(ddv_total)
#
class LossClass(object):

    def __init__(self, u_model, k_model, N_test:int):
        ''' 
        Input:
            u_model: NN for approximating u
            N_test: the number of test functions
        '''
        self.device = device
        self.u_model = u_model
        self.k_model = k_model
        self.getLoss = torch.nn.MSELoss()
        self.GetTest = TestFun(N_test)

    def get_test(self, N_int:int):
        '''
        Input:
            N_int: the number of integral points
        '''
        # Obtain integral points, test function, and gradient of test function
        x_int, w_int = np.polynomial.legendre.leggauss(N_int)
        x_int = torch.tensor(x_int.reshape(-1,1), dtype=dataType)
        w_int = torch.tensor(w_int.reshape(-1,1), dtype=dataType)
        # change of domain: [-1,1] to [0,L]
        w_int = w_int * (ub[0]-lb[0]) / 2. # w_int is scaled to fit domain
        #
        x = x_int*(ub[0]-lb[0])/2. + (ub[0]+lb[0])/2. # Do not forget to transfer x_int into the interval [lb, ub]
        v = self.GetTest.get_v(x_int)
        dv, _ = self.GetTest.get_dv(x_int)
        dv = dv * 2. / (ub[0]-lb[0])# v is also scaled
        
        return x, w_int, torch.tensor(v, dtype=dataType), torch.tensor(dv, dtype=dataType)
    
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
        '''
        N_test, N_int = v.shape[0], v.shape[1]
        ################# The PDE loss
        x = Variable(x.reshape(-1,1), requires_grad=True).to(self.device)
        u = self.u_model(x)
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        #
        kx = self.k_model(x)
        ##a = a_freq.to(x)
        ##fx = (1+torch.sin(a*x)**2)*(kk*np.pi)**2*torch.sin(kk*np.pi*x) - 2*a*kk*np.pi*torch.sin(a*x)*torch.cos(a*x)*torch.cos(kk*np.pi*x)
        fx = 9.81 # constant right hand side
        # The weak form
        #left = torch.sum(kx * du * dv.to(x) * w.to(x), dim=1)
        #right = torch.sum(fx * v.to(x) * w.to(x), dim=1)
        left = torch.sum(kx.unsqueeze(0) * du.unsqueeze(0) * dv.to(x) * w.to(x).unsqueeze(0), dim=1)
        right = torch.sum(fx * v.to(x) * w.to(x).unsqueeze(0), dim=1)

        #print('kx shape:', kx.shape)
        #print('du shape:', du.shape)
        #print('dv shape:', dv.shape)
        #print('w shape:', w.shape)
        #print('left shape:', left.shape)
        #print('right shape:', right.shape)

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
'''
lossClass = LossClass(model_u, model_k, N_test=5)
x, w, v, dv = lossClass.get_test(N_int=100)
#
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(x, v[0], label=r'$v_1$')
axes[0].plot(x, v[1], label=r'$v_2$')
axes[0].plot(x, v[2], label=r'$v_3$')
axes[0].plot(x, v[3], label=r'$v_4$')
axes[0].plot(x, v[4], label=r'$v_5$')
axes[0].set_title('The test functions')
axes[0].legend()
#
axes[1].plot(x, dv[0], label=r'$dv_1$')
axes[1].plot(x, dv[1], label=r'$dv_2$')
axes[1].plot(x, dv[2], label=r'$dv_3$')
axes[1].plot(x, dv[3], label=r'$dv_4$')
axes[1].plot(x, dv[4], label=r'$dv_5$')
axes[1].set_title('The 1st gradients of test functions')
axes[1].legend()
#plt.show()
plt.close()
'''
#############################################################
# Step 4: Train the models
#############################################################
from tqdm import trange
import time
#
epochs = 2500
model_u = MLP_u([1, 200, 60, 60, 200, 1], dtype=torch.float32).to(device)
model_k = MLP_k([1, 200, 60, 60, 200, 1], dtype=torch.float32).to(device)
optimizer = torch.optim.Adam(params=list(model_u.parameters())+list(model_k.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=2/3)
#
w_bd = 1.
w_pde = 1.
w_ob = 5.
############ The training process
error_u_list, error_k_list, u_list, t_list = [], [], [], []
lossClass = LossClass(model_u, model_k, N_test=100)
flag_test = False
for epoch in trange(epochs):
    t0 = time.time()
    loss_bd = lossClass.loss_bd(x_bd, u_bd)
    loss_ob = lossClass.loss_obs(x_obs, u_obs)
    if not flag_test:
        x, w, v, dv = lossClass.get_test(N_int=100)
        flag_test =True
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






# check pre-softplus output of k_model 
raw_outputs = {}
def hook(module, input, output):
    raw_outputs['pre_softplus'] = output.detach()

handle = model_k.net[-1].register_forward_hook(hook)

model_k.eval()
with torch.no_grad():
    x_check = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float32).to(device)
    k_pred = model_k(x_check)

print('x:', x_check.flatten().cpu().numpy())
print('raw (pre-softplus):', raw_outputs['pre_softplus'].flatten().cpu().numpy())
print('k_pred (post-softplus):', k_pred.flatten().cpu().numpy())

handle.remove()
model_k.train()
#--------------------------------

u_query = model_u(x_test.to(device)).detach().cpu()
k_query = model_k(x_test.to(device)).detach().cpu()






u_query = model_u(x_test.to(device)).detach().cpu()
k_query = model_k(x_test.to(device)).detach().cpu()
error_u = torch.sqrt(torch.sum((u_test-u_query)**2)/torch.sum(u_test**2))
error_k = torch.sqrt(torch.sum((k_test-k_query)**2)/torch.sum(k_test**2))

#  Original figure 
fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].plot(x_test, u_query, label='Pred. u')
axes[0].plot(x_test, u_test,  label='True u')
axes[0].set_title(f'$L^2$ relative error for u: {error_u:.4f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u')
axes[0].legend()

axes[1].plot(x_test, k_query, label='Pred. k')
axes[1].plot(x_test, k_test,  label='True k')
axes[1].set_title(f'$L^2$ relative error for k: {error_k:.4f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('k')
axes[1].legend()

axes[2].semilogy(np.cumsum(t_list), error_u_list, label='$L^2$ rel. error (u)')
axes[2].semilogy(np.cumsum(t_list), error_k_list, label='$L^2$ rel. error (k)')
axes[2].set_title('$L^2$ relative error vs. time')
axes[2].set_xlabel('time (s)')
axes[2].set_ylabel('error')
axes[2].legend()

plt.tight_layout()
plt.show()
plt.close()

# ------------------------- new figures -------
pointwise_err_u = torch.abs(u_test - u_query)
pointwise_err_k = torch.abs(k_test - k_query)

# Error vs epoch
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(error_u_list, label='$L^2$ rel. error ($u$)')
ax.semilogy(error_k_list, label='$L^2$ rel. error ($k$)')
ax.set_title(f'Final $L^2$ error: $u = {error_u:.4f}$, $k = {error_k:.4f}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('$L^2$ relative error')
ax.legend()
#ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/error_vs_epoch.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# u(x): predicted, ground truth, pointwise error
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_test, u_query)
ax.set_title(r'Prediction $u_{\theta}(x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/predicted_u.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_test, u_test)
ax.set_title('Ground truth $u(x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/GT_u.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_test, pointwise_err_u)
ax.set_title(r'Pointwise error $|u_{\theta}(x) - u(x)|$')
ax.set_xlabel('$x$')
ax.set_ylabel('error')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/pointwise_error_u.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# k(x): predicted, ground truth, pointwise error
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_test, k_query, color='C1')
ax.set_title(r'Prediction $k_{\theta}(x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$k(x)$')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/predicted_k.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_test, k_test, color='C1')
ax.set_title('Ground truth $k(x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$k(x)$')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/GT_k.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_test, pointwise_err_k, color='C1')
ax.set_title(r'Pointwise error $|k_{\theta}(x) - k(x)|$')
ax.set_xlabel('$x$')
ax.set_ylabel('error')
ax.set_box_aspect(1)
plt.tight_layout()
fig.savefig('Problem_A/pointwise_error_k.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()