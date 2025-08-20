import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from scipy.stats import t
from math import exp, sqrt
import statsmodels.api as sm

# Factorial function
def fac(n):
    if n == 0:
        return 1
    else:
        return n*fac(n-1)

# Solves largest y such that P(Y<=y)<=val where P is a Poisson disturbution for some lambda
def InvPois(val, _lambda_):
    prob = exp(-_lambda_)
    y = 1
    while prob < val:
        prob += (exp(-_lambda_)*_lambda_**y)/fac(y)
        y += 1
    return y-1

def reg_anal(df, pred_months, photo_name):
    # Defining the input variable (x) and the output variable (y)
    x = df['Month ']
    y = df['Qty']

    # Get number of rows
    n = df.shape[0]

    # Get the correlation and the covariance
    #corr = stat.correlation(x,y)
    #cov = stat.covariance(x,y)

    # Plotting a scatter plot of the data points
    plt.scatter(x,y)

    # modeling and fitting the data as poisson regression
    X = sm.add_constant(x)
    pois_model = sm.GLM(y, X, family=sm.families.Poisson())
    pois_results = pois_model.fit()

    # Get the slope/intercept of the linear regression model
    beta_0, beta_1 = pois_results.params

    # Our model is now of the form lambda = exp(beta_0+beta_1*x)
    # finding the predicted lambda at some value
    lambdas = []
    for i in range(pred_months[0],pred_months[1]+1):
        lambdas.append(exp(beta_0+beta_1*i))

    # The value of alpha for confidence interval computation
    alpha = 0.05

    # Obtaining upper and lower bounds for conf interval of the slope
    conf = pois_results.conf_int()
    slope_lower = conf.iloc[1, 0]  # lower bound
    slope_upper = conf.iloc[1, 1]  # upper bound

    # Obtaining the prediction interval for next month
    pred_upper_map = lambda l : max(InvPois(alpha/2, l), InvPois(1-(alpha/2),l))
    pred_lower_map = lambda l : min(InvPois(alpha/2, l), InvPois(1-(alpha/2),l))
    pred_upper = sum(list(map(pred_upper_map,lambdas)))
    pred_lower = sum(list(map(pred_lower_map,lambdas)))

    # Define the points of the line and plotting it
    last_x = int(x.max())
    dom = list(range(0, max(last_x, pred_months[1]) + 1))
    rng = list(map(lambda x : exp(beta_0+beta_1*x), dom))
    plt.plot(dom,rng,color='red')

    # Creating a detailed summary of our regression analysis
    d = {"value" : [beta_1, [round(slope_lower,3), round(slope_upper,3)], beta_0, list(map(lambda x : round(x,3), lambdas)), [round(pred_lower,3), round(pred_upper,3)]]}
    i = ["Beta 1", "95% Confidence Interval", "Beta 0", "Predicted Values", "Prediction Interval"]
    summary = pd.DataFrame(data=d, index=i)

    # Rounding every float to 3 decimal points
    summary['value'] = summary['value'].apply(lambda x : round(x,3) if type(x) is float else x)

    # Saving summary to: summary.csv and summary.tex
    #summary.to_csv('summary.csv')
    #summary.to_latex('summary.tex')

    # Save the graph to the file: laham_betlo_sales_prediction.jpg
    plt.savefig(photo_name, format='jpg', dpi=300)

    # Return the summary
    return summary


