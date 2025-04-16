import pandas as pd

def export_to_csv():
    # import data.xlsx
    data_path = "comp34612/data.xlsx"
    xls = pd.ExcelFile(data_path)
    fm1d = pd.read_excel(xls, 'Follower_Mk1_Dummy')
    fm1 = pd.read_excel(xls, 'Follower_Mk1')
    fm2d = pd.read_excel(xls, 'Follower_Mk2_Dummy')
    fm2 = pd.read_excel(xls, 'Follower_Mk2')
    fm3d = pd.read_excel(xls, 'Follower_Mk3_Dummy')
    fm3 = pd.read_excel(xls, 'Follower_Mk3')
    leader_vars = pd.read_excel(xls, 'Leader Variables')
    test_noises = pd.read_excel(xls, 'Test_Noises')

    # Export to CSV
    fm1.to_csv("fm1.csv", index=False)
    fm2.to_csv("fm2.csv", index=False)
    fm3.to_csv("fm3.csv", index=False)

export_to_csv()