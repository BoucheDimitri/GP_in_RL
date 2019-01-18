import numpy as np
import utils as utl


class BPG:
    
    
    def __init__(self, mdp, policy, theta, alpha, N, M, horizon=20):
        """
        We initialize the parameters of the algorithms.
        """
        self.mdp = mdp
        self.policy = policy
        self.theta = theta
        self.list_theta = [theta]
        self.alpha = alpha
        self.N = N
        self.M = M
        self.n = len(theta)
        self.horizon = horizon
        self.pi = self.policy(*list(self.theta))
        self.policies = [self.pi]
        
        
        ''' Now, we estimate the Fischer information matrix with differents algorithms G1, G2 and G3 '''
    def G1(self, D):
        G = np.zeros((self.U[0].shape[0],self.U[0].shape[0]))
        for path in D:
            for t in range(len(path["states"])):
                g = self.pi.log_gradient(path["states"][t], path["actions"][t]).flatten()
                G += np.tensordot(g,g,0)
        return G/np.sum([len(path["states"]) for path in D])
        
        
    def G2(self):
        G = np.zeros((self.U[0].shape[0],self.U[0].shape[0]))
        for i,u_i in enumerate(self.U):
            i += 1
            G += (1-1/i)*G + np.tensordot(u_i,u_i,0)/i
        return G
    
    
    def G3(self):
        G = np.zeros((self.U[0].shape[0],self.U[0].shape[0]))
        for u_i in self.U:
            G += np.tensordot(u_i,u_i,0)
        return G/self.M
        
        
    def u(self, path):
        ''' We evaluate the score of the path '''
        u_path = np.sum([self.pi.log_gradient(path["states"][t], path["actions"][t]) for t in range(self.horizon)], axis=0)
        return u_path.flatten() 
        
    
    def initialise_BPG(self, model=1):
        R = np.zeros(self.M)
        D = utl.collect_episodes(self.mdp, self.pi, self.horizon, n_episodes=self.M)
        self.U = np.array([self.u(path) for path in D])
        R = np.sum([path["rewards"] for path in D], axis=1)
        G = self.G3()
        self.G_inv = np.linalg.inv(G)
        if model == 1:
            Y = np.array([R[i]*self.U[i] for i in range(self.M)]).T
        else :
            Y = R
        return D, G, Y 
        
        
    def BGP_Eval(self, model=1, sigma=.001):
        ''' We implement the standard BPG Evaluation Algorithm '''
        D, G, Y = self.initialise_BPG(model)
        # Update of K:
        K = np.zeros((self.M,self.M))
        for i,u_i in enumerate(self.U):
            for j,u_j in enumerate(self.U):
                K[i,j] = u_i.dot(self.G_inv.dot(u_j))
        # Update of C, then we compute the posterior mean and covariance:
        if model == 1:
            Z = np.diag(1 + K)
            K = (1 + K)**2
            C = np.linalg.inv(K+sigma**2*np.eye(self.M))
            z0 = 1 + self.n
            return Y.dot(C.dot(Z)), (z0-Z.T.dot(C.dot(Z)))*np.eye(self.n)
        else:
            C = np.linalg.inv(K + sigma**2*np.eye(self.M))
            Z = self.U
            return Z.T.dot(C.dot(Y)), G - Z.T.dot(C.dot(Z))
    
        
    def BGP_Eval_Sparsification(self, model=1, sigma=.001, tau=0):
        ''' Here come a sparse version of the BPG Algorithm '''
        # D contains all the paths
        D, G, Y = self.initialise_BPG(model)
        U_tilde = [self.U[0]]
        z = [1] # worth 1 if the ith path is not in D_tilde
        K_tilde = U_tilde[0].dot(self.G_inv.dot(U_tilde[0]))
        K_tilde_inv = 1/K_tilde
        
        for i in range(1,self.M):
            z = np.hstack((z,0))
            u_path = self.u(D[i])
            k_tilde = np.zeros(len(U_tilde))
            for j,u_j in enumerate(U_tilde):
                k_tilde[j] = u_path.dot(self.G_inv.dot(u_j))
            a = u_path.dot(self.G_inv.dot(u_j))
            if a-np.dot(k_tilde,np.dot(K_tilde_inv,k_tilde))>tau:
                v = np.hstack((k_tilde, a))
                K_tilde = np.vstack((K_tilde, k_tilde))
                K_tilde = np.column_stack((K_tilde,v))
                K_tilde_inv = np.linalg.inv(K_tilde)
                z[-1] = 1
                U_tilde.append(u_path)
        m = sum(z)
        card_D = np.cumsum(z) # card_D[i]=the number of index paths inferior to i in D_tilde
        A = np.zeros((self.M, m))
        for i,j in enumerate(card_D):
            A[i,j-1] = 1
        # Now, we return the expectation and the cov
        if model==1:            
            Z_tilde = np.diag(1 + K_tilde)
            K_tilde = (1 + K_tilde)**2
            mat = A.dot(np.linalg.solve(K_tilde.dot(A.T.dot(A))+sigma**2*np.eye(m),Z_tilde))
            Espectation = Y.dot(mat)
            Cov = (1 + self.n -Z_tilde.dot(A.T.dot(mat)))*np.eye(self.n)
            return Espectation, Cov
        # if model==2:
        Z_tilde = np.squeeze(U_tilde).T
        mat = Z_tilde.dot(np.linalg.solve(A.T.dot(A.dot(K_tilde))+ sigma**2*np.eye(m),A.T))
        Espectation = mat.dot(Y)
        Cov = G - mat.dot(A.dot(Z_tilde.T))
        return Espectation, Cov
    
        
    def theta_update(self, model=1, gradient='regular', sparse=False, sigma=10**-10, tau = 0):
        ''' Here, we update theta '''
        # stepper = Adam() # we don't choose the Adam step but it is a possibility
        # stepper.reset()
        for j in range(self.N):
            n = self.n
            if sparse:
                delta_theta, cov = self.BGP_Eval_Sparsification(model, sigma, tau)
                cov = np.zeros((n,n))
            else:
                delta_theta, cov = self.BGP_Eval(model, sigma)
            if gradient=='regular':
                self.theta = self.theta.copy() + self.alpha[j]*np.dot(np.eye(n)-cov/(n+1),delta_theta) 
                # self.alpha[j]*stepper.update(delta_theta)
            else :
                self.theta = self.theta.copy() + self.alpha[j]*(np.eye(n)-cov/(n+1)).dot(self.G_inv.dot(delta_theta)) 
                # self.alpha[j]*stepper.update(self.G_inv.dot(delta_theta)) 
            self.list_theta.append(self.theta.copy())
            self.pi = self.policy(*list(self.theta))
            self.policies.append(self.pi)

    
    def get_theta(self):
        ''' to get all the values of theta '''
        return np.squeeze(self.list_theta)
    
    
    def get_policies(self):
        ''' to get all the policies associate with theta '''
        return self.policies
        