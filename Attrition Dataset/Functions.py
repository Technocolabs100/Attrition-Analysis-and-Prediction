import pandas as pd
dat=pd.read_csv("D://Kali//TechnoColabs_MiniProject//Attrition Dataset//WA_Fn-UseC_-HR-Employee-Attrition.csv")
#primary key check
def pmc():
    if dat['EmployeeNumber'].nunique()==len(dat['EmployeeNumber']):
        return True
    else :
        return dat[dat.duplicated('EmployeeNumber')]
#list of columns in excel
def colnam():
    col=dat.columns.tolist()
    coldf=pd.DataFrame(col,columns=['Column Names'])
    coldf.to_excel('Column Names.xlsx',index=False)

#function call
print(pmc())
colnam()
