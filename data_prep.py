import pandas as pd

# A function for converting date to number of months from a starting date
def date_to_int(date, start_date):
    lst = date.split('-')
    lst2 = start_date.split('-')
    lst = list(map(lambda x : int(x), lst))
    lst2 = list(map(lambda x : int(x), lst2))
    num = (lst[0]-lst2[0])*12+(lst[1]-lst2[1])
    return num

def data_prep(filename, branch, prod):
    # Reading the datafile
    df = pd.read_csv(filename)

    # Selecting the right data for given branch and product
    df_f = df[(df['Branch'] == branch)&(df['Description'] == prod)]

    # Converting all floats to int
    df_f['Qty'] = df_f['Qty'].apply(lambda x : abs(int(x)))

    # Convert months to numbers
    month_to_date = {'Jan':'2024-01', 'Feb':'2024-02', 'Mar':'2024-03', 'Apr':'2024-04', 'May':'2024-05', 'June':'2024-06', 'July':'2024-07', 'Aug':'2024-08', 'Sep':'2024-09', 'Oct':'2024-10', 'Nov':'2024-11', 'Dec':'2024-12', '2025- Jan':'2025-01', '2025- Feb':'2025-02', '2025- Mar':'2025-03', '2025- Apr':'2025-04', '2025- May':'2025-05', '2025- Jun':'2025-06', '2025- Jul':'2025-07'}
    df_f['Month '] = df_f['Month '].apply(lambda x : month_to_date[x])
    df_f['Month '] = df_f['Month '].apply(lambda x : date_to_int(x,'2024-01'))

    # Returning the final dataframe as the output
    return df_f

#print(pd.read_csv("niamah_sales.csv").columns)
#print(data_prep("niamah_sales.csv","Al Zahra",'AWSAL VEAL MEAT KG'))
