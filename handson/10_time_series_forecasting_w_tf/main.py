# Time series forecasting fundamentals with TensorFlow plus Milestone Project 3: BitPredict

# Experiments:
#   0 Naive model (baseline)
#   1 Dense model, horizon = 1, window = 7
#   2 Same as 1, horizon=1 window=30
#   3 Same as 1, h=7 w=30
#   4 Conv1D h=1 w=7 NA
#   5 LSTM 1 7 NA
#   6 Same as 1 (but w multivariate data) 1 7 Block reward size
#   7 N-BEATs algorithm 1 7 NA
#   8 Ensemble (multiple models optimized on different loss functions) 1 7 NA
#   9 Future prediction model (model to predict future vals, no test data)
#   10 Same as 1 (but with turkey data introduced) 1 7 NA

# Terms / tunable hyperparameters
# horizon: number of timesteps into the future to predict
# window size: number of timesteps to use to predict horizon
import pandas as pd
from matplotlib import pyplot as plt

import bitcoin_pred_pandas as bpp
import naive_model as nm
import model_1_dense as m1
import model_2_dense as m2
import model_3_dense as m3
import model_4_conv1d as m4
import model_5_lstm as m5
import model_6_multivariate as m6
import model_7_n_beats as m7
import model_8_ensemble

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # bpp.run()
    # naive_results = nm.run()
    # m1_results = m1.run()
    # m2_results = m2.run()
    # m3_results = m3.run()
    # m4_results = m4.run()
    # m5_results = m5.run()
    # m6_results = m6.run()
    # m7_results = m7.run()
    m8_results = m8.run()

    # pd.DataFrame({"naive": naive_results["mae"],
    #               "horizon_1_window_7": m1_results["mae"],
    #               "horizon_1_window_30": m2_results["mae"],
    #               "horizon_7_window_30": m3_results["mae"],
    #               "conv1d": m4_results["mae"],
    #               "lstm": m5_results["mae"],
    #               "multivariate": m6_results["mae"],
    #               }, index=["mae"]).plot(figsize=(10, 7), kind="bar")
    # plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
