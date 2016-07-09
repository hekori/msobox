#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# =============================================================================

def rq(A):

    """
    Return a RQ-Decomposition of matrix A.

    Code by Dat Chu and Jan Erik from https://leohart.wordpress.com/
    2010/07/23/rq-decomposition-from-qr-decomposition/.

    Args:
        A: array-like (M, N)
           matrix to be decomposed

    Returns:
        R: array-like (M, M)
           upper triagonal matrix
        Q: array-like (M, N)
           orthogonal matrix

    Raises:
        None
    """

    Q, R = np.linalg.qr(np.flipud(A).T, mode="reduced")
    R = np.flipud(R.T)
    Q = Q.T

    return R[:, ::-1], Q[::-1, :]

# =============================================================================