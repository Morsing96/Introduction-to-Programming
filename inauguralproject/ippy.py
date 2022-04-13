#question 1
import numpy as np
"""defining parameters"""

mp ={'y':1,'p':0.2,'theta':-2} 

def premium(q,mp):
    """ premium policy
    
    Args:
    
        q (numpy.float): Coverage amount
    
        p (float): Probability of the loss being incurred
        
    Returns:
        premium(function): What the agent pays the insurance company
    
    """
    
    return mp['p']*q


def utility(z,mp):
    """ utility function
    
    Args:
    
        z(float): total assets
        mp (dictionary): given parameters
        
    Returns:
       float: utility of assets
        
    
    """
    
    return (z**(1+mp['theta']))/(1+mp['theta'])


def expected_utility(x, q, mp):
    """ Expected utility for insured agent
    
    Args:
    
        x (ndarray): monetary loss
        q (numpy.float): Coverage amount
        mp (dictionary): parameters
        
    Returns:
    
        float: utility of assets for insured agent
    
    """
    z_1 = mp['y'] - x + q - premium(q,mp)
    z_2 = mp['y'] - premium(q,mp)
    return mp['p']*utility(z_1,mp)+(1-mp['p'])*utility(z_2,mp)

#Question 2
mp['x']=0.6

def expected_utility_uninsured(mp):
    """ Expected utility for uninsured agent
    
    Args:
    
        mp (dictionary): parameters
        
    Returns:
    
        float: utility of assets for uninsured agent
    
    """
    
    return mp['p']*utility(mp['y'] - mp['x'],mp)+(1-mp['p'])*utility(mp['y'],mp)

def expected_utility_insured(q, mp, pi):
    """ Expected utility for insured agent where the premium pi is a variable and not a function
    
    Args:
    
        q (numpy.float): Coverage amount
        mp (dictionary): parameters
        pi (numpy.float): What the agent pays the insurance company
        
    Returns:
    
        (numpy.float): utility of assets for insured agent
    
    """

    z_1 = mp['y'] - mp['x'] + q - pi
    z_2 = mp['y'] - pi

    return mp['p']*utility(z_1,mp)+(1-mp['p'])*utility(z_2,mp)
    
    return mp['p']*utility(z_1,mp)+(1-mp['p'])*utility(z_2,mp)


def optimal(pi):
    """ The difference between being insured and not being insured
    
    Args:
    
        mp (dictionary): parameters
        
    Returns:
    
        (numpy float): optimal premium where the expected utility of being insured and not being insured is the same for a given q.
    
    """
    
    return np.absolute(expected_utility_insured(q=0.02, mp=mp, pi=pi)-expected_utility_uninsured(mp))

N=10 #number of elements
q_vector=np.linspace(0.01,0.6,N) # an array of N number of x's equally distributed in the range
pi_vector = np.empty(N)

def optimal_grid(pi):
    """ The difference between being insured and not being insured
    
    Args:
    
        mp (dictionary): parameters
        
    Returns:
    
        (numpy float): optimal premium where the expected utility of being insured and not being insured is the same for grid of qs.
    
    """
    
    return np.absolute(expected_utility_insured(q, mp, pi)-expected_utility_uninsured(mp))

#Question 3
N = 10000
a = 2
b = 7

def MC(a,b,N,gamma,pi):
    """ Monte Carlo simulation drawn from beta distribution
    
    Args:
    
        N (integer): number of draws
        a (integer): parameter
        b (integer): parameter
        
    Returns:
    
        (numpy float): Monte Carlo integration that computes expected utility for given gamma and premium
        
    """
    
    x = np.random.beta(a,b,size=N)
    z_3=mp['y']-(1-gamma)*x-pi
    return np.mean(utility(z_3,mp))

#Question 4
mp['gamma']=0.95

def MC_ins(a,b,N,pi,mp):
    """ Monte Carlo simulation drawn from beta distribution for the insured agents
    
    Args:
    
        N (integer): number of draws
        a (integer): parameter
        b (integer): parameter
        
    Returns:
    
        (numpy float): Monte Carlo integration that computes expected utility for given gamma and premium
        
    """
    
    x = np.random.beta(a,b,N)
    return np.mean(utility(mp['y']-(1-mp['gamma'])*x-pi,mp))


def MC_no(a,b,N,pi,mp):
    """ Monte Carlo simulation drawn from beta distribution for the uninsured agents
    
    Args:
    
        N (integer): number of draws
        a (integer): parameter
        b (integer): parameter
        
    Returns:
    
        (numpy float): Monte Carlo integration that computes expected utility for given gamma and premium
        
    """
    
    x = np.random.beta(a,b,N)
    return np.mean(utility(mp['y']-x,mp))