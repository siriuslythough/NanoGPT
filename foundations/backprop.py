import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        y_fwd = 1/(1+np.exp(-(np.dot(x,w)+b)))
        l = 0.5*((y_fwd-y_true)**2)
        dl_dw = np.round((y_fwd-y_true)*y_fwd*(1-y_fwd)*x,5) 
        dl_db = np.round((y_fwd-y_true)*y_fwd*(1-y_fwd),5)
        return (dl_dw, dl_db)
