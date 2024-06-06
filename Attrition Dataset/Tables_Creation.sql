Create Database AcmeDataset;
Use AcmeDataset;
Select * from FullAttRep;--actual table
---------------------------------------
SELECT EmployeeNumber, Attrition, Age, Gender, MaritalStatus, DistanceFromHome, BusinessTravel,
DailyRate,EnvironmentSatisfaction, JobSatisfaction,RelationshipSatisfaction
Into EmployeePersonalDetails
FROM fullattrep;

Select * from EmployeePersonalDetails;/*table showing  MaritalStatus,
DistanceFromHome, BusinessTravel, DailyRate, EnvironmentSatisfaction,JobSatisfaction,RelationshipSatisfaction*/
------------------------------------------------------------------
Select EmployeeNumber, Age, Attrition,Education, EducationField,Department,JobRole, JobLevel,
JobSatisfaction,TotalWorkingYears,MonthlyIncome,JobInvolvement, RelationshipSatisfaction, EnvironmentSatisfaction
Into EmpEduJobDetails from FullAttRep;

Select * from EmpEduJobDetails;/*table showing EducationField,Department,JobRole,
JobSatisfaction,TotalWorkingYears,MonthlyIncome,JobInvolvement, RelationshipSatisfaction, EnvironmentSatisfaction*/
-------------------------------------------------------------------------------------------------------------------
Select EmployeeNumber, Age, Attrition,JobRole, JobLevel, JobInvolvement, JobSatisfaction, MonthlyIncome
Into EmpJobDetails from FullAttRep;

Select * from EmpJobDetails;/*table showing JobRole, JobLevel, JobInvolvement, JobSatisfaction, MonthlyIncome*/
---------------------------------------------------------------------------------------------------------------
Select EmployeeNumber, Age, Attrition, StandardHours,OverTime, WorkLifeBalance,
PerformanceRating,StockOptionLevel,PercentSalaryHike
into EmpPerformance
from FullAttRep;

Select * from EmpPerformance;/*table showing  StandardHours,OverTime, WorkLifeBalance
PerformanceRating,StockOptionLevel,PercentSalaryHike*/
--------------------------------------------------------------------------------------
Select EmployeeNumber, Age, Attrition, NumCompaniesWorked, TotalWorkingYears,
HourlyRate, DailyRate, MonthlyRate,MonthlyIncome
into EmpExpRate 
from FullAttRep;

Select * from EmpExpRate;/*table showing NumCompaniesWorked, TotalWorkinyears,
HourlyRate, DailyRate, MonthlyRate,MonthlyIncome*/
------------------------------------------------------------------------------
Select EmployeeNumber, Age, Attrition, TrainingTimesLastYear, TotalWorkingYears, MonthlyRate, MonthlyIncome,
YearsAtCompany,YearsSinceLastPromotion
into EmpYearsSalAcme from FullAttRep;

Select * from EmpYearsSalAcme;/*table showing TotalWorkingYears, MonthlyRate, MonthlyIncome, YearsAtCompany,
YearsSinceLastPromotion*/
-------------------------------------------------------------------------------------------------------------
Select EmployeeNumber, Age, Attrition,JobLevel,JobRole,YearsInCurrentRole,YearsAtCompany,
YearsWithCurrManager,JobSatisfaction,
RelationshipSatisfaction, EnvironmentSatisfaction
into EmpWithMgr from FullAttRep;

Select * from EmpWithMgr;/*table showing JobLevel,JobRole,YearsInCurrentRole,
YearsAtCompany,YearsWithCurrManager,JobSatisfaction,
RelationshipSatisfaction, EnvironmentSatisfaction*/
-------------------------------------------------------------------------------
