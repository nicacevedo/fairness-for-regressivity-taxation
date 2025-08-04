import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew
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
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    if isinstance(r, pd.Series):
        r = r.to_numpy()
    y_quants = quantiles_price(y)
    # First metric: F_dev
    f_dev = F_dev(r, y_quants, alpha=alpha)
    groups = get_groups_from_quants(y, y_quants, n_groups)
    f_grp = F_grp(r, groups)
    return {"f_dev":f_dev , "f_grp":f_grp}


def train_test_scatter_plots(
        y_train, y_test, y_pred_train, y_pred_test, 
        n_groups=3, alpha=2, 
        save_plot=False, suffix="", log_scale=False):

    # ratio
    r_pred = y_pred_test / y_test
    r_pred_train = y_pred_train / y_train

    f_metrics = compute_haihao_F_metrics(r_pred, y_test, n_groups=n_groups, alpha=alpha)
    f_dev, f_grp = f_metrics["f_dev"], f_metrics["f_grp"]

    f_metrics_train = compute_haihao_F_metrics(r_pred_train, y_train, n_groups=n_groups)
    f_dev_train, f_grp_train = f_metrics_train["f_dev"], f_metrics_train["f_grp"]




    # Plot real data v/s predicted
    plt.figure(figsize=(8,5))
    plt.scatter(y_test,y_pred_test, facecolor='none', label="test", color="blue")
    plt.plot(y_test, y_test, color="red", label="Real regression")
    plt.legend() # mn_dev={mean_deviation(r_pred):.3f}
    plt.title(f"rmse={np.sqrt(np.mean((y_pred_test - y_test)**2)):.2f} | F_{alpha}dev={f_dev:.3f}, F_{n_groups}grp={f_grp:.3f} | r_std={np.std(r_pred):.3f}, r_skew={skew(r_pred):.3f}")#, kt={kurtosis(r_pred):.3f}")
    if save_plot:
        plt.savefig(f"img/real_vs_pred_test{suffix}.pdf")
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.show()




    plt.figure(figsize=(8,5))
    plt.scatter(y_train,y_pred_train, facecolor='none', label="train", color="green")
    plt.plot(y_train, y_train, color="red", label="Real regression")
    plt.legend() # mn_dev={mean_deviation(r_pred_train):.3f}, 
    plt.title(f"rmse={np.sqrt(np.mean((y_pred_train - y_train)**2)):.2f} | F_{alpha}dev={f_dev_train:.3f}, F_{n_groups}grp={f_grp_train:.3f} | r_std={np.std(r_pred_train):.3f}, r_skew={skew(r_pred_train):.3f}")#, , kt={kurtosis(r_pred_train):.3f}")
    if save_plot:
        plt.savefig(f"img/real_vs_pred_train{suffix}.pdf")
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.show()

    # ratio v/s price
    plt.figure(figsize=(8,5))
    # plt.plot(y_train, y_pred_train, 'o', label="train")
    plt.scatter(y_test,r_pred, facecolor='none', label="test ratio", color="black")
    plt.hlines(1, np.min(y_test), np.max(y_test), alpha=0.7, colors="red", label="Real regression")
    plt.legend()
    plt.title(f"rmse={np.sqrt(np.mean((y_pred_test - y_test)**2)):.2f} | F_{alpha}dev={f_dev:.3f}, F_{n_groups}grp={f_grp:.3f} | r_std={np.std(r_pred):.3f}, r_skew={skew(r_pred):.3f}")#, kt={kurtosis(r_pred):.3f}")
    if save_plot:
        plt.savefig(f"img/real_vs_ratio_test{suffix}.pdf")
    if log_scale:
        plt.xscale("log")
    plt.show()



    # ratio v/s price
    plt.figure(figsize=(8,5))
    # plt.plot(y_train, y_pred_train, 'o', label="train")
    plt.scatter(y_train,r_pred_train, facecolor='none', label="train ratio", color="C2")
    plt.hlines(1, np.min(y_train), np.max(y_train), alpha=0.7, colors="red", label="Real regression")
    plt.legend()
    plt.title(f"rmse={np.sqrt(np.mean((y_pred_train - y_train)**2)):.2f} | F_{alpha}dev={f_dev_train:.3f}, F_{n_groups}grp={f_grp_train:.3f} | r_std={np.std(r_pred_train):.3f}, r_skew={skew(r_pred_train):.3f}")#, , kt={kurtosis(r_pred_train):.3f}")
    if save_plot:
        plt.savefig(f"img/real_vs_ratio_train{suffix}.pdf")
    if log_scale:
        plt.xscale("log")
    plt.show()


