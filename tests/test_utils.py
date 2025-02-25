import utils
import numpy as np



def test_equal():
    # check that obviously unequal unitaries aren't equal
    U1 = np.eye(2)
    U2 = np.array([[1, 0], [0, -1]])
    
    assert not utils.are_equal(U1, U2)
    
    # check that unitaries just offset by a global phase are equal
    
    U1 = np.array([[0, 1], [1, 0]])
    U2 = 1j * U1
    
    assert utils.are_equal(U1, U2)