import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        #convert all input lists to numpy arrays, 
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        #forward pass
        z1 = np.dot(W1,x) + b1
        l1_out = np.maximum(0.0,z1)
        pred = np.dot(W2,l1_out)+b2
        #compute loss
        loss = np.mean((pred-y_true)**2)
        #grad wrt pred
        dl_dpred = 2*(pred-y_true)/len(y_true)
        #grad in layer 2
        dW2 = np.outer(dl_dpred,l1_out)
        db2 = dl_dpred
        #grad wrt layer 1 output
        dl_dl1_out = np.dot(W2.T, dl_dpred)
        #element wise relu to get gradient wrt relu and input passed to relu
        dl_dz1 = dl_dl1_out*(z1>0)
        #gradient in layer 1
        dW1=np.outer(dl_dz1, x)
        db1 = dl_dz1

        return {
            'loss': float(np.round(loss, 4)),
            'dW1': dW1.tolist(),
            'db1': db1.tolist(),
            'dW2': dW2.tolist(),
            'db2': db2.tolist()
        }
