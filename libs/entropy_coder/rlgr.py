"""This module supplies functions for adaptive run-length golomb-rice (RLGR) coding.
"""
import numpy as np


def rlgr(R, return_bitstream=False):
    """RLGR computes number of bits to code R with Adaptive Run Length Golomb Rice

    Args:
      R: Nx1 array of integers to be coded
      return_bitstream: True to return a bitstream, false otherwise

    Returns:
      bitStreamCount: number of bits needed to code R
      bitStream: ceil(bitCount/8)x1 uint8 array containing bit stream
    """
    # Length of input.
    N = len(R)

    # Constants.
    L = 4
    U0 = 3
    D0 = 1
    U1 = 2
    D1 = 1
    quotientMax = 24

    # Initialize state.
    k_P = 0
    k_RP = 10 * L
    bitStreamCount = 0  # number of bits written to bitStream

    # Preprocess data from signed to unsigned.
    U = 2 * R
    neg = (R < 0)
    Uneg = -U[neg]
    U[neg] = Uneg - 1

    # Allocate space for bitstream if caller requests it.
    if return_bitstream:
        bitStream = np.zeros(max(100, 2 * N), np.uint8)
        byteStreamCount = 0
        bitBuffer = np.zeros(1, np.uint64)
        bitBufferCount = 0
        bitBufferCountMax = 64

    # Process data one sample at a time (time consuming in python).
    n = 0
    while n < N:

        k = k_P // L
        k_RP = min(k_RP, 31 * L)
        k_R = k_RP // L
        pow2k_R = 2**k_R

        u = U[n]  # symbol to encode

        if k == 0:  # no-run mode

            # Output GR code for symbol u.
            # bits = bits + gr(u,k_R)
            quotient = u // pow2k_R  # number of 1s to write
            if quotient < quotientMax:
                # 'quotient' 1s + 0 + k_R-bit remainder
                bitStreamCount = bitStreamCount + (quotient + 1 + k_R)
            else:
                # 'quotientMax' 1s + 32-bit value of u
                bitStreamCount = bitStreamCount + quotientMax + 32

            # Write to bitstream if caller requests it.
            if return_bitstream:
                if quotient < quotientMax:
                    remainder = u - quotient * pow2k_R
                    bitPattern = np.bitwise_or(
                        np.left_shift(np.left_shift(
                            np.int64(1), quotient) - 1, 1 + k_R),
                        np.int64(remainder))
                    bitPatternCount = quotient + 1 + k_R
                else:
                    bitPattern = np.bitwise_or(
                        np.left_shift(np.left_shift(
                            np.int64(1), quotientMax) - 1, 32),
                        np.int64(u))
                    bitPatternCount = quotientMax + 32
                bitBuffer = np.bitwise_or(
                    np.left_shift(bitBuffer, bitPatternCount), bitPattern)
                bitBufferCount = bitBufferCount + bitPatternCount
                if bitBufferCount > bitBufferCountMax:
                    # Overflow error, not recoverable.
                    bitStreamCount = -1
                    bitStream = np.zeros(0, np.uint8)
                    return (bitStreamCount, bitStream)
                while bitBufferCount >= 8:
                    bitBufferCount = bitBufferCount - 8
                    bitStream[byteStreamCount] = np.bitwise_and(
                        np.right_shift(bitBuffer, bitBufferCount), 255)
                    byteStreamCount = byteStreamCount + 1
                if bitStreamCount != 8 * byteStreamCount + bitBufferCount:
                    print('error at n=#d\n', n)

            # Adapt k_R.
            p = u // (2**k_R)  # number of probability halvings
            if p == 0:
                k_RP = max(0, k_RP - 2)
            elif p > 1:
                k_RP = k_RP + p + 1

            # Adapt k.
            if u == 0:
                k_P = k_P + U0
            else:  # u > 0
                k_P = max(0, k_P - D0)

        else:  # k > 0  # run mode

            # m = 2**k = expected length of run of zeros
            m = np.left_shift(1, k)

            # Parse off up to m symbols,
            # through first non-zero symbol,
            # counting number of zero symbols before it.
            zeroCount = 0
            while u == 0:
                zeroCount = zeroCount + 1
                if zeroCount >= m or n >= N - 1:
                    break
                n = n + 1
                u = U[n]
            # At this point, either u>0 or (u=0 & (zeroCount>=m | n>=N-1).
            # That is, either u>0 or (u=0 & zeroCount>=m) or (u=0 & n>=N-1).
            if zeroCount == m:
                # Found a complete run of zeroCount = m zeros.
                # Output a 0.
                bitStreamCount = bitStreamCount + 1

                # Write to bitstream if caller requests it.
                if return_bitstream:
                    bitPattern = 0
                    bitPatternCount = 1
                    bitBuffer = np.bitwise_or(
                        np.left_shift(bitBuffer, bitPatternCount), bitPattern)
                    bitBufferCount = bitBufferCount + bitPatternCount
                    while bitBufferCount >= 8:
                        bitBufferCount = bitBufferCount - 8
                        bitStream[byteStreamCount] = np.bitwise_and(
                            np.right_shift(bitBuffer, bitBufferCount), 255)
                        byteStreamCount = byteStreamCount + 1
                    if bitStreamCount != 8 * byteStreamCount + bitBufferCount:
                        print('error at n=#d\n', n)

                # Adapt k.
                k_P = k_P + U1
            else:  # zeroCount < m, and either u>0 or (u=0 and n>=N-1)
                # Found a partial run of zeroCount < m zeros.
                if u > 0:
                    # Partial run ended normally with a non-zero symbol u.
                    # Output a 1 + length of partial run + GR code for non-zero symbol.
                    # bits = bits + 1 + k + gr(u-1,k_R)
                    quotient = (u - 1) // pow2k_R  # number of 1s to write
                    if quotient < quotientMax:
                        bitStreamCount = bitStreamCount + \
                            1 + k + (quotient + 1 + k_R)
                    else:
                        bitStreamCount = bitStreamCount + 1 + k + quotientMax + 32

                    # Write to bitstream if caller requests it.
                    if return_bitstream:
                        bitPattern = np.bitwise_or(m, zeroCount)
                        bitPatternCount = 1 + k
                        bitBuffer = np.bitwise_or(
                            np.left_shift(bitBuffer, bitPatternCount), bitPattern)
                        bitBufferCount = bitBufferCount + bitPatternCount
                        while bitBufferCount >= 8:
                            bitBufferCount = bitBufferCount - 8
                            bitStream[byteStreamCount] = np.bitwise_and(
                                np.right_shift(bitBuffer, bitBufferCount), 255)
                            byteStreamCount = byteStreamCount + 1
                        if quotient < quotientMax:
                            remainder = (u - 1) - quotient * pow2k_R
                            bitPattern = np.bitwise_or(
                                np.left_shift(
                                    np.left_shift(np.int64(1), quotient) - 1, 1 + k_R),
                                np.int64(remainder))
                            bitPatternCount = quotient + 1 + k_R
                        else:
                            bitPattern = np.bitwise_or(
                                np.left_shift(
                                    np.left_shift(np.int64(1), quotientMax) - 1, 32),
                                np.int64(u))
                            bitPatternCount = quotientMax + 32
                        bitBuffer = np.bitwise_or(
                            np.left_shift(bitBuffer, bitPatternCount), bitPattern)
                        bitBufferCount = bitBufferCount + bitPatternCount
                        if bitBufferCount > bitBufferCountMax:
                            # Overflow error, not recoverable.
                            bitStreamCount = -1
                            bitStream = np.zeros(0, np.uint8)
                            return
                        while bitBufferCount >= 8:
                            bitBufferCount = bitBufferCount - 8
                            bitStream[byteStreamCount] = np.bitwise_and(
                                np.right_shift(bitBuffer, bitBufferCount), 255)
                            byteStreamCount = byteStreamCount + 1
                        if bitStreamCount != 8 * byteStreamCount + bitBufferCount:
                            print('error at n=#d\n', n)

                    # Adapt k_R.
                    p = (u - 1) // (2**k_R)  # number of probability halvings
                    if p == 0:
                        k_RP = max(0, k_RP - 2)
                    elif p > 1:
                        k_RP = k_RP + p + 1

                    # Adapt k.
                    k_P = max(0, k_P - D1)
                else:  # u = 0 and n = N-1
                    # Partial run ended with a zero symbol, at end of sequence.
                    # Output a 0.  Leave it to decoder to know number of symbols needed.
                    bitStreamCount = bitStreamCount + 1

                    # Write to bitstream if caller requests it.
                    if return_bitstream:
                        bitPattern = 0
                        bitPatternCount = 1
                        bitBuffer = np.bitwise_or(
                            np.left_shift(bitBuffer, bitPatternCount), bitPattern)
                        bitBufferCount = bitBufferCount + bitPatternCount
                        while bitBufferCount >= 8:
                            bitBufferCount = bitBufferCount - 8
                            bitStream[byteStreamCount] = np.bitwise_and(
                                np.right_shift(bitBuffer, bitBufferCount), 255)
                            byteStreamCount = byteStreamCount + 1
                        if bitStreamCount != 8 * byteStreamCount + bitBufferCount:
                            print('error at n=#d\n', n)

        n = n + 1

    # Flush state to bitstream if caller requests it.
    if return_bitstream:
        if bitBufferCount > 0:
            bitStream[byteStreamCount] = np.bitwise_and(
                np.left_shift(bitBuffer, 8 - bitBufferCount), 255)
            byteStreamCount = byteStreamCount + 1
        bitStream = bitStream[:byteStreamCount]
        return (bitStreamCount, bitStream)

    return bitStreamCount


def irlgr(bitStream, N):
    """IRLGR decodes bitStream into integers using Adaptive Run Length Golomb Rice.

    Args:
      bitStream: uint8 array to be decoded
      N: number of symbols to reconstruct

    Returns:
      R: Nx1 array of decoded signed integers
    """
    # Constants.
    L = 4
    U0 = 3
    D0 = 1
    U1 = 2
    D1 = 1
    quotientMax = 24

    # Initialize state.
    k_P = 0
    k_RP = 10 * L
    bitStreamCount = 0  # number of bits read from bitStream
    bitBufferCount = 0  # number of undecoded bits in bitBuffer
    bitBuffer = np.zeros(1, np.uint64)  # initial value of bitBuffer
    bitStreamCountMax = 8 * len(bitStream)
    bitBufferCountMax = 64

    # Allocate space for decoded unsigned integers.
    U = np.zeros(N, np.int32)

    # Process data one sample at a time (time consuming in Matlab).
    n = 0
    while n < N:

        k = k_P // L
        k_RP = min([k_RP, 31 * L])
        k_R = k_RP // L
        pow2k_R = 2**k_R

        # Load up bitBuffer.
        while bitBufferCount <= bitBufferCountMax - 8 and bitStreamCount <= bitStreamCountMax - 8:
            bitBuffer = np.bitwise_or(
                np.left_shift(bitBuffer, 8),
                np.uint64(bitStream[bitStreamCount // 8]))
            bitStreamCount = bitStreamCount + 8
            bitBufferCount = bitBufferCount + 8

        if k == 0:  # no-run mode

            # Read floor(u/(2^k_R) + 1 + k_R bits for next symbol u>=0.

            # Read in quotient = floor(u/pow2k_R) 1s before a zero.
            quotient = 0
            while quotient < quotientMax and ((bitBuffer >>
                                               (bitBufferCount - 1)) & 1) == 1:
                quotient = quotient + 1
                bitBufferCount = bitBufferCount - 1
            while bitBufferCount <= bitBufferCountMax - 8 and bitStreamCount <= bitStreamCountMax - 8:
                bitBuffer = np.bitwise_or(
                    np.left_shift(bitBuffer, 8),
                    np.uint64(bitStream[bitStreamCount // 8]))
                bitStreamCount = bitStreamCount + 8
                bitBufferCount = bitBufferCount + 8

            if quotient < quotientMax:
                # Read in the 0.
                bitBufferCount = bitBufferCount - 1

                # Read in k_R bits, containing the remainder u - floor(u/pow2k_R).
                u = quotient * pow2k_R + np.bitwise_and(
                    np.right_shift(bitBuffer, bitBufferCount - k_R), pow2k_R - 1)
                bitBufferCount = bitBufferCount - k_R
            else:
                # Read in 32 bits, containing u.
                assert 32 - bitBufferCount >= 0
                u = np.bitwise_and(
                    np.left_shift(bitBuffer, 32 - bitBufferCount), 2**32 - 1)
                bitBufferCount = bitBufferCount - 32

            # Output the decoded symbol u >= 0.
            U[n] = u
            n = n + 1

            # Adapt k_R.
            p = u // (2**k_R)  # number of probability halvings
            if p == 0:
                k_RP = max(0, k_RP - 2)
            elif p > 1:
                k_RP = k_RP + p + 1

            # Adapt k.
            if u == 0:
                k_P = k_P + U0
            else:  # u > 0
                k_P = max(0, k_P - D0)
        else:  # k > 0 # run mode

            # m = 2^k = expected length of run of zeros
            m = np.left_shift(1, k)

            # Read in next bit.
            if ((bitBuffer >> (bitBufferCount - 1)) & 1) == 0:
                bitBufferCount = bitBufferCount - 1

                # Bit is 0, which means there is a complete run of m zeros.

                # Output the decoded zeros.
                while m > 0 and n < N:
                    U[n] = 0
                    n = n + 1
                    m = m - 1

                # Adapt k.
                k_P = k_P + U1
            else:  # bitvalue(bitBuffer,bitBufferCount) == 1
                bitBufferCount = bitBufferCount - 1

                # Bit is 1, which means there is a partial run of zeroCount < m zeros.

                # First read in k bits to specify zeroCount.
                assert bitBufferCount - k >= 0
                zeroCount = np.bitwise_and(
                    np.right_shift(bitBuffer, bitBufferCount - k), m - 1)
                bitBufferCount = bitBufferCount - k

                # Output the decoded zeros.
                while zeroCount > 0:
                    U[n] = 0
                    n = n + 1
                    zeroCount = zeroCount - 1

                # Then read in floor((u-1)/(2^k_R) + 1 + k_R bits for next symbol u >= 1.

                # Read in quotient = floor((u-1)/pow2k_R) 1s before a zero.
                quotient = 0
                while quotient < quotientMax and ((bitBuffer >>
                                                   (bitBufferCount - 1)) & 1) == 1:
                    quotient = quotient + 1
                    bitBufferCount = bitBufferCount - 1
                while bitBufferCount <= bitBufferCountMax - 8 and bitStreamCount <= bitStreamCountMax - 8:
                    bitBuffer = np.bitwise_or(
                        np.left_shift(bitBuffer, 8),
                        np.uint64(bitStream[bitStreamCount // 8]))
                    bitStreamCount = bitStreamCount + 8
                    bitBufferCount = bitBufferCount + 8

                if quotient < quotientMax:
                    # Read in the 0.
                    bitBufferCount = bitBufferCount - 1

                    # Read in k_R bits, containing the remainder (u-1) - floor((u-1)/pow2k_R).
                    u = 1 + quotient * pow2k_R + np.bitwise_and(
                        np.right_shift(bitBuffer, bitBufferCount - k_R), pow2k_R - 1)
                    bitBufferCount = bitBufferCount - k_R
                else:
                    # Read in 32 bits, containing u.
                    assert 32 - bitBufferCount >= 0
                    u = np.bitwise_and(
                        np.left_shift(bitBuffer, 32 - bitBufferCount), 2**32 - 1)
                    bitBufferCount = bitBufferCount - 32

                # Output the decoded symbol u >= 1.
                U[n] = u
                n = n + 1

                # Adapt k_R.
                p = (u - 1) // (2**k_R)  # number of probability halvings
                if p == 0:
                    k_RP = max(0, k_RP - 2)
                elif p > 1:
                    k_RP = k_RP + p + 1

                # Adapt k.
                k_P = max(0, k_P - D1)

    # Postprocess data from unsigned to signed.
    R = np.zeros(N, np.int32)
    even = (np.mod(U, 2) == 0)
    R[even] = U[even] / 2
    R[~even] = -(U[~even] + 1) / 2
    return R
