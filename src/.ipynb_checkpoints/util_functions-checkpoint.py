import numpy as np

# Metrics

def mean_deviation(x, base_level=1):
    return np.mean(np.abs(x - 1))



# Haihao's metrics



def positive_part(x):
    x[x < 0] = 0
    return x

def quantiles_price(y):
    m = y.size
    y_tilde = y.copy()
    for i,x in enumerate(y):
        y_tilde[i] = np.sum(x >= y) # N_x
    return y_tilde / m



def F_dev(r, y_quants, alpha=2, base_level=1):
    w1 = lambda x: np.exp(-alpha * x)
    w2 = lambda x: np.exp(-alpha * (1-x))
    sum_r_1_pos = positive_part(r - base_level) @ w1(y_quants)
    sum_1_r_pos = positive_part(base_level - r) @ w2(y_quants)
    
    return sum_r_1_pos + sum_1_r_pos


def get_groups_from_quants(y, y_quants, n_groups=3):
    bounds = np.linspace(0,1,n_groups+1)
    # print(bounds[1:])
    groups = dict()
    if n_groups > 1:
        lb = bounds[0]
        for i,ub in enumerate(bounds[1:]):
            # print(lb, ub)

            groups[i] = np.where((lb < y_quants) & (y_quants <= ub))[0]
            lb = ub # update bound
        # print(groups)
    else:
        groups[0] = np.where(y_quants <= float("inf"))[0]

    return groups
    # groups = {}
    # for i, y_i in enumerate(y):


def F_grp(r, groups):
    score = 0
    for g1 in groups:
        for g2 in groups:
            if g1 < g2:
                m_g1 = len(groups[g1])
                m_g2 = len(groups[g2])
                score+=1/m_g1/m_g2*np.sum(positive_part(r[groups[g1], np.newaxis] - r[groups[g2]]))
                # for i in groups[g1]:
                #     score+= np.sum(positive_part(r[i] - r[groups[g2]]))
                # print(groups[g1])
                # print(groups[g2])
                # positive_part(r[0] - r[1])
                # print([positive_part(r[i] - r[j]) for i in groups[g1] for j in groups[g2]])
                # print(np.sum([positive_part(r[i] - r[j]) for i in groups[g1] for j in groups[g2]]))
                # score+= 1/m_g1/m_g2*np.sum([positive_part(r[i] - r[j]) for i in groups[g1] for j in groups[g2]])
    return score


def compute_haihao_F_metrics(r, y, n_groups=3, alpha=2):
    # Get the quantiles
    y_quants = quantiles_price(y.to_numpy())
    # First metric: F_dev
    f_dev = F_dev(r, y_quants, alpha=alpha)
    groups = get_groups_from_quants(y.to_numpy(), y_quants, n_groups)
    f_grp = F_grp(r.to_numpy(), groups)
    return {"f_dev":f_dev , "f_grp":f_grp}


