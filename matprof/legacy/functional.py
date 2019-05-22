import numpy as np


def MP_1nnDist_pair(DL, DR, SubseqLength, ZNORM):
    if SubseqLength > len(DL) / 2:
        raise Exception('Error: Time series DL is too short relative to desired subsequence length')

    if SubseqLength > len(DR) / 2:
        raise Exception('Error: Time series DR is too short relative to desired subsequence length')
    if DL.shape[0] < DL.shape[1]:
        DL = DL[:, np.newaxis]

    D = np.array([
        [DL],
        [DR]
    ])
    L_start = 1    # 0 ?
    L_end = DL.shape[1]
    R_start = DL.shpae[1] + 1
    R_end = D.shape[1]
    L_domain = np.arange(L_start, L_end - SubseqLength+1)
    R_domain = np.arange(R_start, R_end - SubseqLength+1)

    MatrixProfileLength = D.shape[1] - SubseqLength+1
    Dist = np.kron(np.ones(MatrixProfileLength, 1), np.inf)
    Index = np.zeros((MatrixProfileLength, 1))
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = fastfindNNPre(D, SubseqLength)

    pickedIdx = np.arange(0, MatrixProfileLength)
    dropval = 0
    distanceProfile = np.zeros(MatrixProfileLength, 0)
    lastz = np.zeros(MatrixProfileLength, 0)
    updatePos = np.full([MatrixProfileLength, 1], False)

    for i in np.arange(0, MatrixProfileLength):
        idx = pickedIdx[i]
        subsequence = D[idx:idx + SubseqLength-1]

        if i == 0:
            distanceProfile[:, 0],  lastz, dropval, lastsumy, lastsumy2 = \
                fastfindNN(X, subsequence, n, SubseqLength, sumx2, sumx, meanx, sigmax2, sigmax)
            distanceProfile[:, 0] = np.abs(distanceProfile)
            firstz = lastz
        else:
            lastsumy = lastsumy - dropval + subsequence[-1]
            lastsumy2 = lastsumy2 - dropval ** 2 + subsequence[-1] ** 2
            meany = lastsumy /  SubseqLength
            sigmay2 = lastsumy2 / SubseqLength**2
            sigmay = np.sqrt(sigmay2)

            lastz[1:n-SubseqLength+1] = \
                lastz[0:n-SubseqLength] - D[0:n-SubseqLength] * dropval + D[SubseqLength+1:n] * subsequence[SubseqLength]

            lastz[0] = firstz[i]
            if ZNORM:
                distanceProfile[:, 0] = np.sqrt(2 * (SubseqLength - (lastz - SubseqLength * meanx * meany) / (sigmax * sigmay)))
            else:
                distanceProfile[:, 0] = np.sqrt(sumx2 + lastsumy2 - 2 * lastz)

            dropval = subsequence[0]

        if idx <= L_end - SubseqLength + 1:
            exclusionZone = np.arange(L_start, L_end)
        elif idx >= R_start:
            exclusionZone = R_domain

        distanceProfile[exclusionZone] = np.inf

        updatePos[:] = False

        if idx <= L_end - SubseqLength + 1:
            updatePos[R_domain] = distanceProfile[R_domain] < Dist[R_domain]
            Dist[updatePos] = distanceProfile[updatePos]
            Index[updatePos] = idx - L_end
        elif idx >= R_start:
            updatePos[L_domain] = distanceProfile[L_domain] < Dist[L_domain]
            Dist[updatePos] = distanceProfile[updatePos]
            Index[updatePos] = idx

        if np.mod(i, 1000) == 0:
            print(i)

    LDist = Dist[L_start:L_end - SubseqLength + 1]
    LIndex = Index[L_start:L_end - SubseqLength + 1]

    RDist = Dist[R_start:R_end - SubseqLength + 1]
    RIndex = Index[R_start:R_end - SubseqLength + 1]

    return LDist, RDist, LIndex, RIndex


def fastfindNNPre(x, m):
    n = len(x)  # x.shape[1]

    x[n+1: 2*n] = 0
    X = np.fft.fft(x)

    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(x ** 2)

    tmp1 = np.array([
        [0],
        [cum_sumx2[0:n-m]]
    ])

    sumx2 = cum_sumx2[m:n] - tmp1

    tmp2 = np.array([
        [0],
        [cum_sumx[0:n-m]]
    ])

    sumx = cum_sumx[m:n] - tmp2

    meanx = sumx / m

    sigmax2 = (sumx2 / m) - (meanx ** 2)
    sigmax = np.sqrt(sigmax2)

    return X, n, sumx2, sumx, meanx, sigmax2, sigmax


def fastfindNN(X, y, n, m, sumx2, sumx, meanx, sigmax2, sigmax):
    global TSC_SUBSEQ_get_ZNORM = None
    dropval = y[0]
    y = y[::-1]
    y[m + 1:2 * n] = 0

    Y = np.fft.fft(y)
    Z = X * Y
    z = np.fft.ifft(Z)

    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y, 2))
    meany = sumy / m
    sigmay2 = sumy2 / m - meany ** 2
    sigmay = np.sqrt(sigmay2)

    if not TSC_SUBSEQ_get_ZNORM and TSC_SUBSEQ_get_ZNORM:
        dist2 = 2 * (m - z[m:n] - m * meanx * meany) / (sigmax * sigmay)
    else:
        dist2 = sumx2 * sumy2 - 2 * z[m:n]

    dist = np.sqrt(dist2)
    lastz = np.real(z[m:n])

    return dist, lastz, dropval, sumy, sumy2
