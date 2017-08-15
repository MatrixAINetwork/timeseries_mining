import numpy as np


def dist_subsequence(subsequence,
                     wholeseries,
                     position_increment,
                     dist_func,
                     dist_func_params):
    """
    Compute the distance between subsequence and wholeseries.
    The distance is the minimum local distance between wholeseries and subsequence.
    The sliding window size depend on the length of subsequence.
    
    Parameters:
    --------------
    :param subsequence: sub sequence, list or array 
    :param wholeseries: whole series , list or array
    :param position_increment: the increment of position
    :param dist_func:  distance function, function call
    :param dist_func_params:  distance function parameters
    :return: 
        distance, float value, 
    """
    assert len(subsequence) < len(wholeseries),\
        "sub sequence should be shorter than the whole series. (len(subsequence), len(timeseries)), (%s, %s)"\
        % (len(subsequence), len(wholeseries))

    dist_min = np.inf
    spLen = len(subsequence)
    tsLen = len(wholeseries)
    pos = 0
    while pos <= tsLen - spLen:
        dist_current = dist_func(subsequence, wholeseries[pos:(pos + spLen)], **dist_func_params)
        if dist_current < dist_min:
            dist_min = dist_current
        pos += position_increment
    return dist_min

######################################################
"""
    p-distance:
        euclidean : p = 2
        manhattan : p = 1
        infinity : p = infinity
"""
######################################################


def euclidean(ts1, ts2, cut_value=np.Inf):
    """
    Euclidean distance calculation, complexity: O(n)
    
    :param ts1: 
    :param ts2: 
    :param cut_value: 
    :return: 
    """

    assert ts1.shape == ts2.shape, "shape didn't match, " + ts1.shape + "," + ts2.shape

    n = len(ts1)
    dist = 0
    for i in range(n):
        dist += (ts1[i] - ts2[i])**2
        if dist >= cut_value:
            return cut_value

    return dist


def manhattan(ts1, ts2, cut_value=np.Inf):
    """
    Manhattan Distance calculation, complexity: O(n)
    
    :param ts1: 
    :param ts2: 
    :param cut_value: 
    :return: 
    """
    assert ts1.shape == ts2.shape, "shape didn't match, " + ts1.shape + "," + ts2.shape

    n = len(ts1)
    dist = 0
    for i in range(n):
        dist += np.abs(ts1[i] - ts2[i])
        if dist >= cut_value:
            return np.Inf
    return dist


def infinity(ts1, ts2, cut_value=np.Inf):
    """
    infinity norm distance calculation, complexity: O(n)
    
    :param ts1: 
    :param ts2: 
    :param cut_value: 
    :return: 
    """
    assert ts1.shape == ts2.shape, "shape didn't match, " + ts1.shape + "," + ts2.shape
    n = len(ts1)
    dist_max = 0
    for i in range(n):
        temp = np.abs(ts1[i] - ts2[i])
        if temp > dist_max:
            dist_max = temp
        if dist_max >= cut_value:
            return np.Inf

    return dist_max


######################################################
"""
    elastic distance, derived DTW:
        basic dtw
        speeded dtw
        weighted_dtw
        
"""
######################################################


def dtw_basic(ts1, ts2, cut_value=np.Inf):
    """
    basic DTW distance calculation, complexity: O(nm)
    :param ts1: 
    :param ts2: 
    :param cut_value: 
    :return: 
    """
    len1 = len(ts1)
    len2 = len(ts2)

    # dp[i,j] indicate the distance between ts1[0,...,i-1] and ts2[0,....,j-1]
    # the length of ts1[0,...,i-1] is i
    # the length of ts2[0,....,j-1] is j
    dp = np.zeros((len1 + 1, len2 + 1), dtype='float')
    dp[0, 0] = 0  # the distance between two zero length sequence is 0
    for j in range(1, len2 + 1):
        dp[0][j] = np.inf
    for i in range(1, len1 + 1):
        dp[i][0] = np.inf

    # find the path has minimum distance
    is_overflow = True
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + (ts1[i - 1] - ts2[j - 1]) ** 2
            if is_overflow and dp[i][j] < cut_value:
                is_overflow = False
        if is_overflow:
            return np.Inf

    return dp[len1][len2]


def dtw_win(ts1, ts2, win, cut_value=np.Inf):
    """
    speed up DTW.
    this works under the assumption that it is unlikely for qi and cj to be matched if i and j are too far apart.
    The threshold is determined by a window size 'win'.
    speed up the inner loop.
    -------------------
    :param ts1: 
    :param ts2: 
    :param win: 
    :param cut_value:
    :return: 
    """
    assert win >= 1, "the win parameter should be a value larger than 0, win=" + str(win)

    len1 = len(ts1)
    len2 = len(ts2)

    win = max(win, abs(len1 - len2))  # the second item is the length of tail
    dp = np.zeros((len1 + 1, len2 + 1)) + np.inf
    dp[0, 0] = 0
    is_overflow = True
    for i in range(1, len1 + 1):
        for j in range(max(1, i - win), min(len2 + 1, i + win)):
            # speed up, just check the j within win scope
            dist = (ts1[i - 1] - ts2[j - 1]) ** 2
            dp[i][j] = dist + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
            if is_overflow and dp[i][j] < cut_value:
                is_overflow = False
        if is_overflow:
            return np.Inf

    return dp[len1][len2]


def dtw_weighted(ts1, ts2, g=0, cut_value=np.Inf):
    """
    paper: Weighted dynamic time warping for time series classification, Jeong Young Seon, 2011
    :param ts1: 
    :param ts2: 
    :param g: 
    :param cut_value: 
    :return: 
    """

    WEIGHT_MAX = 1.0

    def calculate_weight(series_length, penalty):
        weights = np.zeros(series_length)
        half_length = float(series_length) / 2.0
        for i in range(1, series_length+1):
            weights[i-1] = WEIGHT_MAX / (1.0 + np.exp(-penalty * (i - half_length)))
        return weights

    ints1 = ts1.copy()
    ints2 = ts2.copy()

    # insure ints1 larger than ints2
    if len(ints1) < len(ints2):
        temp = ints1
        ints1 = ints2
        ints2 = temp

    len1 = len(ints1)
    len2 = len(ints2)

    weights = calculate_weight(len1, penalty=g)  # initialization
    dp = np.zeros([len1+1, len2+1])
    dp[0][0] = 0
    for i in range(1, len1+1):
        dp[i][0] = np.Inf
    for j in range(1, len2+1):
        dp[0][j] = np.Inf

    is_overflow = True
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            dp[i][j] = weights[np.abs(j-i)] * (ints1[i - 1] - ints2[j - 1]) ** 2
            dp[i][j] += min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            if is_overflow and dp[i][j] < cut_value:
                is_overflow = False
        if is_overflow:
            return np.inf

    return dp[len1][len2]

######################################################

######################################################


def LBKeogh(ts1, ts2, win, cut_value=np.Inf):
    """
    complexity: O(n)
    :param ts1: 
    :param ts2: 
    :param win: 
    :param cut_value:
    :return: 
    """
    LBSum = 0
    for ind, value in enumerate(ts1):
        lower_bound = min(ts2[(ind - win if ind >= win else 0):(ind + win)])
        upper_bound = max(ts2[(ind - win if ind >= win else 0):(ind + win)])

        if value > upper_bound:
            LBSum = LBSum + (value - upper_bound) ** 2
        elif value < lower_bound:
            LBSum = LBSum + (value - lower_bound) ** 2

        if LBSum >= cut_value:
            return np.Inf

    return LBSum

######################################################
"""
    Longest Common Subsequence (LCSS):
    paper: Indexing Multi-Dimensional Time-Series with Support for Multiple Distance Measures, Vlachos, 2003
    paper: Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching, Transactions, 2009

"""
######################################################


def lcss(ts1, ts2, epsilon, cut_value=np.Inf):
    """
    complexity: O(m*n), m=len(ts1), n=len(ts2)
    
    delta function didn't realization
    
    :param ts1: 
    :param ts2: 
    :param epsilon: range = (0,1)
    :param cut_value: 
    :return: 
    """

    len1 = len(ts1)
    len2 = len(ts2)

    status = np.zeros([len1+1, len2+1])
    UP = 1
    Left = 2
    Up_left = 0
    # status[i][j] == 1 up
    # status[i][j] == 2 left
    # status[i][j] == 0 up-left

    lcss = np.zeros([len1+1, len2+1])  # lcss[i][j] indicate the lcss between ts1[0,...,i-1] and ts2[0,....,j-1]

    for i in range(len1+1):
        lcss[i][0] = 0
    for j in range(len2+1):
        lcss[0][j] = 0

    is_overflow = True
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if abs(ts1[i-1] - ts2[j-1]) <= epsilon:
                # match
                lcss[i][j] = lcss[i-1][j-1] + 1
                status[i][j] = Up_left
            elif lcss[i-1][j] >= lcss[i][j-1]:
                lcss[i][j] = lcss[i-1][j]
                status[i][j] = UP
            else:
                lcss[i][j] = lcss[i][j-1]
                status[i][j] = Left
            if is_overflow and lcss[i][j] < cut_value:
                is_overflow = False
        if is_overflow:
            return np.Inf

    verbose = False
    if verbose:
        str_head = "ts2\t\t"
        for i in range(len2):
            str_head = str_head + str(ts2[i]) + '\t'
        print(str_head)

        for i in range(len1+1):
            if i == 0:
                str_print = "ts1"
            else:
                str_print = str(ts1[i-1])
            for j in range(len2+1):
                str_print = str_print + '  ' + str(status[i][j]) + "({})".format(str(lcss[i][j]))
            print(str_print)

    return 1 - lcss[len1][len2]/min(len1, len2)  # Note: the larger lcss the smaller similarity measure should be




######################################################
"""
    Time warp edit:
    paper: Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching, Transactions, 2009
"""
######################################################


def twe(a, b, lambda_, v, tsa=None, tsb=None, cut_value=np.Inf):

    lena = len(a)
    lenb = len(b)

    if tsa is None:
        tsa = np.array(range(lena))
    if tsb is None:
        tsb = np.array(range(lenb))

    dp = np.zeros([lena+1, lenb+1]) + np.Inf
    dp[0][0] = 0

    is_overflow = True
    for i in range(1, lena+1):
        for j in range(1, lenb+1):
            # match
            if i > 1 and j > 1:
                dist1 = dp[i-1][j-1] + (a[i-1]-b[j-1])**2 + (a[i-2]-b[j-2])**2 + \
                        v*abs(tsa[i-1] - tsb[i-1])*abs(tsa[i-2] - tsb[i-2])
            else:
                dist1 = dp[i-1][j-1] + (a[i-1]-b[j-1])**2 + v*abs(tsa[i-1]-tsb[i-1])

            # delete a
            if i > 1:
                dist2 = dp[i-1][j] + (a[i-1] - a[i-2])**2 + v*abs(tsa[i-1]-tsa[i-2]) + lambda_
            else:
                dist2 = dp[i-1][j] + abs(a[i-1]) + lambda_

            # delete b
            if j > 1:
                dist3 = dp[i][j-1] + (b[i-1] - b[j-2])**2 + v*abs(tsb[j-1] - tsb[j-2]) + lambda_
            else:
                dist3 = dp[i][j-1] + abs(b[j-1]) + lambda_

            dp[i][j] = min(dist1, dist2, dist3)
            if is_overflow and dp[i][j] < cut_value:
                is_overflow = False
        if is_overflow:
            return np.Inf

    return dp[lena][lenb]


######################################################
"""
    move-split-merge (MSM)
    paper: The move-split-merge metric for time series, Stefan , 2013

"""
######################################################


def msm(tsa, tsb, c, cut_value=np.Inf):
    def penalty(new_point, x, y):
        if x <= new_point <= y or x >= new_point >= y:
            return c
        else:
            return c + min(abs(new_point - x), abs(new_point - y))

    lena = len(tsa)
    lenb = len(tsb)
    dp = np.zeros([lena, lenb])
    dp[0][0] = abs(tsa[0] - tsb[0])

    for i in range(1, lena):
        dp[i][0] = dp[i-1][0] + penalty(tsa[i], tsa[i - 1], tsb[0])
    for j in range(1, lenb):
        dp[0][j] = dp[0][j-1] + penalty(tsb[j], tsa[0], tsb[j - 1])

    is_overflow = True
    for i in range(1, lena):
        for j in range(1, lenb):
            dist1 = dp[i-1][j-1] + abs(tsa[i] - tsb[j])  # match
            dist2 = dp[i-1][j] + penalty(tsa[i], tsa[i - 1], tsb[j])  # delete a
            dist3 = dp[i][j-1] + penalty(tsb[j], tsa[i], tsb[j - 1])  # delete b
            dp[i][j] = min(dist1, dist2, dist3)
            if is_overflow and dp[i][j] < cut_value:
                is_overflow = False
        if is_overflow:
            return np.Inf

    return dp[lena-1][lenb-1]


######################################################
"""
    Complexity invariant distance (CID):
    paper: The move-split-merge metric for time series, Stefan , 2013

"""
######################################################


def cid(ts1, ts2, distfunc, distfunc_params=None, cut_value=np.Inf):
    if distfunc_params is None:
        distfunc_params = {}
    dist = distfunc(ts1, ts2, **distfunc_params)

    len1 = len(ts1)
    len2 = len(ts2)
    dist1 = 0
    for i in range(len1-1):
        dist1 += (ts1[i] - ts1[i+1])**2
    dist2 = 0
    for i in range(len2-2):
        dist2 += (ts2[i] - ts2[i+1])**2

    return dist * (max(dist1, dist2) / min(dist1, dist2))

######################################################
"""
    Derivative DTW (ddtw):

"""
######################################################


def ddtw(ts1, ts2, alpha, distfunc, distfunc_params=None, diffn=1, cut_value=np.Inf):
    diff1 = np.diff(ts1, diffn)
    diff2 = np.diff(ts2, diffn)

    if distfunc_params is None:
        distfunc_params = {}

    dist1 = distfunc(ts1, ts2, **distfunc_params)
    dist2 = distfunc(diff1, diff2, **distfunc_params)

    dist = alpha*dist1 + (1-alpha)*dist2

    return dist

######################################################
"""
    Derivative transform distance (DTDC):

"""
######################################################


def dtdc(ts1, ts2, alpha, beta, distfunc, distfunc_params=None, diffn=1, cut_value=np.Inf):
    def cos(ts):
        tsr = np.zeros(len(ts))
        for i in range(len(ts)):
            tsr[i] = 0
            for j in range(len(ts)):
                tsr[i] += np.cos((np.pi/2) * (j-1/2) * (i-1))
        return tsr

    diff1 = np.diff(ts1, diffn)
    diff2 = np.diff(ts2, diffn)
    cos1 = cos(ts1)
    cos2 = cos(ts2)

    if distfunc_params is None:
        distfunc_params = {}
    dist1 = distfunc(ts1, ts2, **distfunc_params)
    dist2 = distfunc(diff1, diff2, **distfunc_params)
    dist3 = distfunc(cos1, cos2, **distfunc_params)

    return alpha*dist1 + beta*dist2 + (1-alpha-beta)*dist3


DISTANCE_FUNCTION = {'euclidean': euclidean,
                     'manhattan': manhattan,
                     'infinity': infinity,
                     'dtw_basic': dtw_basic,
                     'dtw_win': dtw_win,
                     'dtw_weighted': dtw_weighted,
                     'ddtw': ddtw,
                     'LBKeogh': LBKeogh,
                     'lcss': lcss,
                     'twe': twe,
                     'msm': msm,
                     'cid': cid,
                     'dtdc': dtdc}

