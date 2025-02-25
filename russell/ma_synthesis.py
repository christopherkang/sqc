from .int_field import ZetaInt, ZSqrt2Int
from . import int_field, utils
import numpy as np

X = np.array([
        [ZetaInt((0, 0, 0, 0)), ZetaInt((1, 0, 0, 0))],
        [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))], 
    ])
    
Y = np.array([
    [ZetaInt((0, 0, 0, 0)), ZetaInt((0, 0, -1, 0))],
    [ZetaInt((0, 0, 1, 0)), ZetaInt((0, 0, 0, 0))], 
])

Z = np.array([
    [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
    [ZetaInt((0, 0, 0, 0)), ZetaInt((-1, 0, 0, 0))], 
])

HSq2 = np.array([
    [ZetaInt((1, 0, 0, 0)), ZetaInt((1, 0, 0, 0))],
    [ZetaInt((1, 0, 0, 0)), ZetaInt((-1, 0, 0, 0))],
])

S = np.array([
    [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
    [ZetaInt((0, 0, 0, 0)), ZetaInt((0, 0, 1, 0))],
])

T = np.array([
    [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))], 
    [ZetaInt((0, 0, 0, 0)), ZetaInt((0, 1, 0, 0))],
])

GATE_LOOKUP = {
    "H": HSq2,
    "S": S,
    "T": T,
    "X": X,
    "Y": Y,
    "Z": Z,
}

FLOAT_GATE_LOOKUP = {
    "H": np.array([
        [1, 1],
        [1, -1],
    ]) / np.sqrt(2),
    "S": np.array([
        [1, 0],
        [0, 1j],
    ]),
    "T": np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)],
    ]),
    "X": np.array([
        [0, 1],
        [1, 0],
    ]),
    "Y": np.array([
        [0, -1j],
        [1j, 0],
    ]),
    "Z": np.array([
        [1, 0],
        [0, -1],
    ]),
}

def count_sqrt2_factors_old(x: ZetaInt):
    """Counts the number of sq2 factors in a ZetaInt value

    Args:
        x (ZetaInt): Input value

    Returns:
        int: Number of sq2 factors
    """
    if x == ZetaInt((0, 0, 0, 0)):
        return np.inf
    k = 1
    sq2 = ZetaInt((0, 1, 0, -1))
    
    while (x / sq2)[1] == ZetaInt((0, 0, 0, 0)):
        # can increase power
        k += 1
        sq2 = sq2 * ZetaInt((0, 1, 0, -1))
    
    return k - 1

def count_sqrt2_factors(x: ZetaInt):
    """Counts the number of sq2 factors in a ZetaInt value

    Args:
        x (ZetaInt): Input value

    Returns:
        int: Number of sq2 factors
    """
    if x == ZetaInt((0, 0, 0, 0)):
        return np.inf
    # if z is divisible by sqrt(2)^k
    # then z**2 is divisble by 2^k
    
    xsq = x * x
    pwrs = [_max_pow_of_2(v) for v in xsq]
    return np.min(pwrs)

def _max_pow_of_2(n: int):
    # return the maximum power of 2 that divides n
    if n == 0:
        return np.inf
    
    k = 0
    while n % 2 == 0:
        k += 1
        n = n // 2
    
    return k

def get_T_count(M, k):
    # we are given a unitary U = M / sqrt(2**k)
    # confirm that this is indeed unitary
    U = [[x.eval() for x in row] for row in M] / np.sqrt(2**k)
    traceval = np.trace(U @ U.conj().T)
    
    assert np.isclose(np.abs(traceval), 2), f"M/sqrt(2**k) is not unitary; tr={traceval} M={M}, k={k}"
    
    # now, note that for M = T we would have that k = 0
    # and yet U (xX + yY + zZ) U^\dag has a sqrt(2) factor
    # similarly, for U with $c$ T factors, we would have
    # U (xX + yY + zZ) U^\dag = 1/\sqrt{2}^c
    # furthermore, U = M/sqrt(2)**k so really
    # M (xX + yY + zZ) M^\dagger = sqrt(2)**(2k - c)
    # so count the number of sqrt 2 factors F = 2k - c
    # so that c = 2k - F
    # we are getting F' = 2k - c + 2 because of the 2x overhead
    # so c = 2k - F' + 2
    
    so3_form = so3_from_u2(M, leader=2) # this will be 2x the so3 form
    
    # check the number of sq2 factors in each element
    all_sq2_factors = np.array([[count_sqrt2_factors(x) for x in row] for row in so3_form])
    total_sq2_factors = np.max(all_sq2_factors) - 1
    
    c = 2 * k - total_sq2_factors + 2
    return c
    

def get_xyz(M, leader=1):
    return np.array([np.trace(M @ X), np.trace(M @ Y), np.trace(M @ Z)]) * leader / 2

def so3_from_u2(U, leader=1):
    # U will be a 2x2 matrix
    # we want to find \hat{U} in SO(3)
    # where \hat{U} (x, y, z) = (x', y', z')
    # such that U (xX + y Y + z Z) U^\dag = x' X + y' Y + z' Z
    
    # use one-hot vectors and trace to achieve this
    
    Udag = U.T
    Udag = [[~x for x in row] for row in Udag]
    
    hat_U = np.zeros((3, 3), dtype=object)
    hat_U[:, 0] = get_xyz(U @ X @ Udag, leader=leader)
    hat_U[:, 1] = get_xyz(U @ Y @ Udag, leader=leader)
    hat_U[:, 2] = get_xyz(U @ Z @ Udag, leader=leader)
    
    return hat_U

def get_xyz_int(M):
    return np.array([np.trace(M @ FLOAT_GATE_LOOKUP["X"]), np.trace(M @ FLOAT_GATE_LOOKUP["Y"]), np.trace(M @ FLOAT_GATE_LOOKUP["Z"])]) / 2

def so3_from_u2_float(U):
    Udag = U.conj().T
    hat_U = np.zeros((3, 3), dtype=float)
    
    ## TODO: all of this can be optimized
    hat_U[:, 0] = get_xyz_int(U @ FLOAT_GATE_LOOKUP["X"] @ Udag)
    hat_U[:, 1] = get_xyz_int(U @ FLOAT_GATE_LOOKUP["Y"] @ Udag)
    hat_U[:, 2] = get_xyz_int(U @ FLOAT_GATE_LOOKUP["Z"] @ Udag)
    
    return hat_U
    
    

def _parity(elem: ZetaInt):
    return elem[0] % 2

def _M_parity(M):
    return np.array([[_parity(x) for x in row] for row in M])

def _M_gcd(M):
    gcd = M[0][0]
    for row in M:
        for elem in row:
            gcd = int_field.gcd(gcd, elem)
    
    return gcd

def _normal_M(M):
    gcd = _M_gcd(M)
    return [[elem // gcd for elem in row] for row in M]

def conj_transpose(M):
    out = M.T
    out = [[~x for x in row] for row in out]
    return out

# Lemma 4.10
peel_selection = {
    0: "HT",
    1: "SHT",
    2: "T",
}

def peel_lemma(M):
    # M is SO3 representation
    assert M.shape == (3, 3)
    # lemma 4.10 dictates what's next
    # we need to check whether the matrix has an all zeros row
    parity_M = _M_parity(M)
    
    for row_idx in range(3):
        if all([x == 0 for x in parity_M[row_idx]]):
            # we have found a row with all zeros
            # we can now peel
            # we can now peel the row
            return peel_selection[row_idx]
    
    raise ValueError(f"No row with all zeros found, {M}")

def num_zero_rows(M):
    assert M.shape == (3, 3)
    num_zero_rows = 0
    
    for row in M:
        if all([x == 0 for x in row]):
            num_zero_rows += 1
    
    return num_zero_rows

def eval_M(M):
    return np.array([[x.eval() for x in row] for row in M])


def count_sqrt2_factors_matrix(M):
    return int(np.min([[count_sqrt2_factors(x) for x in row] for row in M]))

def reduce(M):
    # reduce the SO3 form until there is just a single row of zero elements
    assert not np.allclose(eval_M(M), np.zeros((3, 3))), "Matrix is already zero"
    
    while num_zero_rows(_M_parity(M)) >= 2:
        M = M // ZetaInt((0, 1, 0, -1))
    
    return M

# check if matrix is diagonal
def is_diagonal(M):
    return np.allclose(M, np.diag(np.diag(M)))


def get_int_so3(U):
    so3_int = so3_from_u2_float(U)
    
    # turn matrix into int_matrix
    so3_int = np.rint(so3_int).astype(int)
    return so3_int

def eval_word_float(word):
    out_U = np.eye(2)
    
    for g in word:
        out_U = out_U @ FLOAT_GATE_LOOKUP[g]
    
    return out_U


def decompose_clifford(U):
    # consumes a Unitary U in float basis using the Selinger Ross clifford_of_so3 technique
    
    assert U.shape == (2, 2), "Matrix must be 2x2"
    
    seq = []
    
    clifford_lookup = {
        "H": (2, [1, 0, 0]),
        "HX": (2, [-1, 0, 0]),
        "SH": (2, [0, 1, 0]),
        "SHX": (2, [0, -1, 0]),
        "X": (2, [0, 0, -1]),
        "S": (1, [-1, 0, 0]),
        "SS": (1, [0, -1, 0]),
        "SSS": (1, [1, 0, 0]),
    }
    
    while True:
        so3 = get_int_so3(U)
        if np.allclose(so3, np.eye(3)):
            return seq
        
        potential_seq = []
        for key, (col_idx, so3_col) in clifford_lookup.items():
            if np.allclose(so3[:, col_idx], so3_col):
                potential_seq.append(key)
        
        assert len(potential_seq) >= 1, f"No potential seq found for {U}, {so3}"
        # apply the first matching potential_seq
        seq.append(potential_seq[0])
        U = eval_word_float(potential_seq[0]).conj().T @ U


def synthesize(M):
    # we will synthesize a unitary which can be written as $U = M / sqrt{2^k}$
    
    curr_U = M
    
    # identify the k value:
    col_sum = np.abs(M[0, 0].eval())**2 + np.abs(M[1, 0].eval())**2
    # this needs to equal to some 2**k where U = M / sqrt{2^k}
    k = np.log2(col_sum)
    
    seq = []
    iters = 1
    
    keep_decomposing = True
    
    while True:
        # if there was one T, the representation would have a 1/sqrt{2} factor
        # so we should multiply by sqrt{2}
        # if the matrix is scaled by c, the representation is scaled by c**2
        # thus, we need to divide the representation by c**2
        # and multiply it by sqrt(2)**number of Ts
        
        # normalize the SO3 matrix
        SO3 = so3_from_u2(curr_U, leader=2)
        sq2_factors = count_sqrt2_factors_matrix(SO3)
        SO3 = SO3 // ZetaInt((0, 1, 0, -1))**sq2_factors
        
        # if sq2_factors > 2:
        #     curr_U = curr_U // (ZetaInt((0, 1, 0, -1))**((sq2_factors - 2) // 2))
        
        # exit condition is that SO3 is one-hot row and column wise
        so3_eval = eval_M(SO3)
        num_nonzero = np.count_nonzero(so3_eval) == 3
        row1_nonzero = np.count_nonzero(so3_eval[0]) == 1
        col1_nonzero = np.count_nonzero(so3_eval[:, 0]) == 1
        row2_nonzero = np.count_nonzero(so3_eval[1]) == 1
        col2_nonzero = np.count_nonzero(so3_eval[:, 1]) == 1
        
        nonzeros = np.array([num_nonzero, row1_nonzero, col1_nonzero, row2_nonzero, col2_nonzero])
        if np.all(nonzeros):
            break
        
        iters += 1
        # print(f"Iteration {iters}")
        if iters > 200:
            assert 1 == 0, "Infinite loop"
        
        peel_result = peel_lemma(SO3)
        seq.append(peel_result)
        
        if peel_result == "SHT":
            curr_U = conj_transpose(S @ HSq2 @ T) @ curr_U
        elif peel_result == "HT":
            curr_U = conj_transpose(HSq2 @ T) @ curr_U
        elif peel_result == "T":
            curr_U = conj_transpose(T) @ curr_U
        else:
            raise ValueError("Peel lemma failed")
    
    # now, decompose the remaining clifford component
    curr_U = eval_M(curr_U)
    rescale_factor = np.trace(curr_U.conj().T @ curr_U) / 2
    
    curr_U = curr_U / np.sqrt(rescale_factor)
    assert utils.are_equal(curr_U.conj().T @ curr_U, np.eye(2)), curr_U
    
    clif_component = decompose_clifford(curr_U)
    
    return seq + clif_component
    