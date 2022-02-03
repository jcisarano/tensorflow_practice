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


import bitcoin_pred_pandas as bpp
import naive_model as nm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # bpp.run()
    nm.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/