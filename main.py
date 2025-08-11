from reg_anal import reg_anal
from data_prep import data_prep

branch = input("Enter branch: ")

product = input("Enter product: ")

period_1 = int(input("Enter period: from: "))

period_2 = int(input("Enter priod: to: "))

period = [period_1, period_2]

df = data_prep("niamah_sales.csv", branch, product)

summ = reg_anal(df,period,'prediction.jpg')

summ.to_csv('summary.csv')

print(f'''



Here, the predicted increase of sales after one month for {product} at branch {branch} is: {summ.loc['Beta 1']}. We are 95% confident that predicted increse of sales after one month is in the range {summ.loc['95% Confidence Interval']}. Also, the predicted quantity sold in the months {period} is {summ.loc['Predicted Values']}, respectively. We are 95% confident that the total predicted value for these months is in the range {summ.loc['Prediction Interval']}.

    A detailed summary csv file is uploaded in the current directory, as well as a visual graph for the predicted model.''')
