import numpy as np
import parameters as P

from int_field import ZetaInt, ZSqrt2Int
import ma_synthesis as MA

import utils

mat_eval = lambda M: np.array([[x.eval() for x in row] for row in M])


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
symb_eval = lambda x, y, z: x * X + y * Y + z * Z


def test_so3_from_su2():
    # use some established test cases
    TEST_CASES = {
        "I": (
            np.array([
            [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
            [ZetaInt((0, 0, 0, 0)), ZetaInt((1, 0, 0, 0))]
            ]), 
            np.array([
            [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
            [ZetaInt((0, 0, 0, 0)), ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
            [ZetaInt((0, 0, 0, 0)), ZetaInt((0, 0, 0, 0)), ZetaInt((1, 0, 0, 0))]])
            ),
        "X": (
            np.array([
            [ZetaInt((0, 0, 0, 0)), ZetaInt((1, 0, 0, 0))],
            [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))]
            ]), 
            np.array([
            [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
            [ZetaInt((0, 0, 0, 0)), ZetaInt((-1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
            [ZetaInt((0, 0, 0, 0)), ZetaInt((0, 0, 0, 0)), ZetaInt((-1, 0, 0, 0))]])
            ),
    }
    
    for test_case, (U, true_so3) in TEST_CASES.items():
        U = np.array(U)
        true_so3 = np.array(true_so3)
        
        # we can now evaluate the matrix
        # we will do this by comparing the matrices elementwise
        assert np.allclose(mat_eval(MA.so3_from_u2(U) - true_so3), 0)

def test_get_T_count():
    word = "HTHTHTHT"
    U = _synthesize_unitary_from_word(word)
    k = word.count("H")
    
    computed_T_count = MA.get_T_count(U, k)
    
    assert computed_T_count == word.count("T"), (computed_T_count, U, k, word.count("T"))


# def test_so3_from_su2_rigorous():
#     # randomly generate xX + yY + zZ matrix
#     x = ZetaInt((np.random.randint(-10, 10), 0))
#     y = ZetaInt((np.random.randint(-10, 10), 0))
#     z = ZetaInt((np.random.randint(-10, 10), 0))
    
    
    
#     U = symb_eval(x, y, z)
#     Udag = U.T
#     Udag = [[~x for x in row] for row in Udag]
    
#     hat_U = MA.so3_from_u2(U)
    
#     prime_vec = hat_U @ np.array([x, y, z]).T
#     prime_U = symb_eval(*prime_vec)
    
#     assert np.allclose(mat_eval(U @ prime_U @ Udag), mat_eval(prime_U)), (U, prime_U, Udag)

def _synthesize_unitary_from_word(word):
    U = np.array([
        [ZetaInt((1, 0, 0, 0)), ZetaInt((0, 0, 0, 0))],
        [ZetaInt((0, 0, 0, 0)), ZetaInt((1, 0, 0, 0))]
    ])
    k = word.count("H")
    
    for gate in word:
        U = U @ MA.GATE_LOOKUP[gate]
        
    ueval = mat_eval(U) / np.sqrt(2)**k
    assert np.allclose(ueval @ ueval.conj().T, np.eye(2)), ueval
    # now, check the synthesis output
    
    return U

def test_decompose_clifford():
    # randomly create a unitary using Cliffords 
    # e.g. X, Y, Z, H, S
    
    for _ in range(P.MANY_TRIALS):
        num_syllables = 10
        word = list(np.random.choice(["X", "Y", "Z", "H", "S"], num_syllables))
        word = "".join(word)
        
        
        U = MA.eval_word_float(word)
        
        es = MA.decompose_clifford(U)
        es = "".join(es)
        
        recovered_U = MA.eval_word_float(es)
        assert utils.are_equal(U, recovered_U), f"word={word} es={es}"


def test_synthesis():
    # randomly create a unitary using S, H, T:
    for _ in range(P.MANY_TRIALS):
        num_syllables = 10
        word = list(np.random.choice(["SHT", "HT"], num_syllables))
        word = "".join(word)
        
        # word = "HTTH"
        U = _synthesize_unitary_from_word(word)
        
        es = MA.synthesize(U)
        es = "".join(es)
        
        assert es == word, f"word={word} es={es}"

def test_synthesis_rigorous():
    # randomly create a unitary using any clifford+T sequence
    for _ in range(P.FEW_TRIALS):
        num_gates = 100
        word = list(np.random.choice(["X", "Y", "Z", "H", "S", "T"], num_gates))
        word = "".join(word)
        
        U = _synthesize_unitary_from_word(word)
        
        es = MA.synthesize(U)
        es = "".join(es)
        
        # now, simply check that the unitaries are equal
        U_float = MA.eval_word_float(word)
        recovered_U = MA.eval_word_float(es)
        assert utils.are_equal(U_float, recovered_U), f"word={word} es={es}"


def test_count_sq2_factors():
    for _ in range(P.MANY_TRIALS):
        # randomly generate a zeta_int with some k powers of sq(2)
        
        sq2 = ZetaInt((0, 1, 0, -1))
        power = np.random.randint(1, 20)
        num = sq2**power
        
        assert power == MA.count_sqrt2_factors(num)
        