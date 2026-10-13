# %% [markdown]
# # Assignment 4
# Name: <font color=blue>______文杰______</font>,       ID: <font color=blue>______123090612______</font>

# %%
import numpy as np
import matplotlib.pylab as plt

# %% [markdown]
# ### Helpful functions
# The following functions may be helpful for your coding, but you can also choose to code it by yourself or modify it.

# %% [markdown]
# #### plot the two figures

# %%
def plot1(numbers, lam):
    '''For problem 5-2
        Parameters:
        ----------
        numbers: a list or an array with type int
            collections of the total number of arrivals for each round
        lam : float
            rate
    '''
    p = lambda k: np.exp(-lam)*lam**k/np.math.factorial(k)
    x = list(range(0, np.max(numbers)+1))
    plt.figure()
    plt.hist(numbers, range=(0, np.max(numbers)+1), bins=np.max(numbers)+1, alpha = 0.5, rwidth=0.5, label='empirical', color='orange', density=True)
    plt.bar(x, [p(k) for k in x], alpha = 0.5, width=0.5, label='theoretic')
    plt.xlabel(f'the number of arrival on [0, {time}]')
    plt.ylabel('probability')
    plt.legend(loc='upper left')
    plt.show()

# %%
def plot2(first_arrivals, lam, bins:int):
    '''For problem 5-3
        Parameters:
        ----------
        first_arrivals: a list or an array with type float
            collections of the first arrival time for each round
        lam: float
            rate
        bins: int 
            the number of bins of the histogram
    '''
    bin_width = time / bins
    G = lambda t: np.exp(-lam*t) # tail prob
    hight = lambda left: (G(left) - G(left+bin_width)) / bin_width 
    x = np.linspace(0, time, num=bins, endpoint=False)
    plt.figure()
    plt.hist(first_arrivals, range=(0, time), bins=bins, alpha = 0.5, rwidth=0.5, label='empirical', color='orange', density=True)
    plt.bar(x, [hight(k) for k in x], alpha = 0.5, width=0.5*bin_width, label='theoretic')
    plt.legend(loc='upper right')
    plt.xlabel('the time of the first arrival')
    plt.ylabel('intensity')
    plt.show()

# %% [markdown]
# Note that the intensity has the relation with the frequency:
# $$\text{intensity}\times \text{bin width} = \text{frequency}$$

# %% [markdown]
# If you want to sample from some distributions, you can use `np.random`. For example
# - exponential random variable with rate $\lambda$ can be sampled by `np.random.exponential(1/lambda)` (see [docs](https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html))
# - poisson random variable with mean $\lambda$ can be sampled by  `np.random.poisson(lambda)` (see [docs](https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html))
# - uniform random variable with range [low, high] can be sampled by  `np.random.uniform(low,high,size=n)` (see [docs](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html))

# %% [markdown]
# ## Question 5

# %%
time = 1 # we just focus on the interval [0,1] in this problem.
np.random.seed(42) # set random seed for reproduction.

# %% [markdown]
# ### 5-1
# Write two functions **Poisson1** and **Poisson2** to simulate the sequence of arrival times of events on $[0,1]$ for a Poisson process with rate $\lambda$, where $\lambda$ is the input to your functions. In **Poisson1**, the simulation is based on generating i.i.d. inter-arrival times. In **Poisson2**, you first generate the total number of arrivals and then the conditional distribution of arrival times.

# %%
def Poisson1(lam):
    '''
        Parameters:
        ----------
        lam : float
            rate

        Returns
        -------
        A list or an array
            arrival times on interval [0,time].
    '''
    raise NotImplementedError # Please delete this line after finishing your code

# %%
def Poisson2(lam):
    '''
        Parameters:
        ----------
        lam : float
            rate

        Returns
        -------
        A list or an array
            arrival times on interval [0,time].
    '''
    raise NotImplementedError # Please delete this line after finishing your code

# %% [markdown]
# ### 5-2
# Set $\lambda =10$. Run **Poisson1** for 10000 rounds, record the total number of arrivals in each round. Plot the empirical distribution of the simulated number of arrivals, and validate your codes by comparing the empirical distribution with the theoretical distribution.

# %%
lam = 10
N = 10000

# %%


# %% [markdown]
# ### 5-3
# Run **Poisson2** for 10000 rounds, record the first arrival time. (What if there is no arrival on $[0,1]$?) Plot the empirical distribution of the first arrival time, and validate your codes by comparing the empirical distribution with the theoretical distribution.

# %%
lam = 10
N = 10000

# %%



