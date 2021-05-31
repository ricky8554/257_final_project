import numpy as np

# mdp = PI()
class PI(object):
    def __init__(self, env, gamma=0.99, desc=None):
        P,nS,nA = env.P, env.nS, env.nA
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.P = np.zeros((nS, nA, nS))
        self.L = np.zeros((nS, nA))    
        self.Y = np.full(nS, np.inf)   
        for s in range(nS):
            for a in range(nA):
                for (pr, ns, re, done) in P[s][a]:
                    self.P[s, a, ns] += pr
                    self.L[s, a] -= re*pr
                    if done:
                        self.Y[ns] = 0


    def policy_iteration(self):
    
        num_iter = 100
        term_sta = np.isfinite(self.Y)
        ntrm_sta = ~term_sta
        ntrm_I = np.eye(np.sum(ntrm_sta))
        iall_sta = np.arange(self.nS)
        pi = np.zeros((num_iter+1, self.nS), dtype='int')
        self.maxQPI = np.zeros(( self.nS), dtype='float')
        self.minQPI = np.zeros(( self.nS), dtype='float')
        Vpi = np.zeros((num_iter+1, self.nS))
        Vpi[:, term_sta] = self.Y[term_sta]  
        last = 0

        for k in range(num_iter):
            Ppi = self.P[iall_sta, pi[k]]
            A = ntrm_I - self.gamma * Ppi[ntrm_sta, :][:, ntrm_sta]
            b = self.L[iall_sta, pi[k]][ntrm_sta] + \
                Ppi[ntrm_sta, :][:, term_sta] @ self.Y[term_sta]
            Vpi[k, ntrm_sta] = np.linalg.solve(A, b)
            Qpi = self.L[ntrm_sta, :] + self.gamma * \
                np.sum(self.P[ntrm_sta, :, :] * Vpi[k, None, None, :], axis=2)
            
            self.maxQPI[ntrm_sta] = np.max(Qpi, axis=1)
            self.minQPI[ntrm_sta] = np.min(Qpi, axis=1)
            pi[k+1, ntrm_sta] = np.argmin(Qpi, axis=1)

            if np.array_equal(pi[k+1], pi[k]):
                last = k+1
                break
        
         
        
        return pi[last]
    def get_QPI(self):
        return self.minQPI,self.maxQPI