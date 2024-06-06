Use AcmeDataset;
With Attr As (Select * from EmployeePersonalDetails where Attrition=1 and EnvironmentSatisfaction>=3)
Select Attr.EmployeeNumber,Attr.Age, Attr.DistanceFromHome,Attr.BusinessTravel,
EmpExpRate.DailyRate, EmpExpRate.HourlyRate, EmpExpRate.MonthlyRate, EmpExpRate.MonthlyIncome,
EmpExpRate.NumCompaniesWorked As 'PrevJobNo.', EmpExpRate.TotalWorkingYears as 'WorkEx' 
From Attr 
Join EmpExpRate on Attr.EmployeeNumber=EmpExpRate.EmployeeNumber;
/*The result shows us the relation among various variables which need to be explored further.
to do so I intend to make dynamic functions that can tell us functions particulars affecting attrition.
Furthermore the relation among DialyRate, HourlyRate, MonthlyRate and MonthlyIncome will be studied independently
hereon through tables EmpPerformnace and EmpExpRate*/
------------------------------------------------------------------------------------------------------------------

With Attr As (Select * from EmployeePersonalDetails where Attrition=1 and EnvironmentSatisfaction>=3)
Select Attr.EmployeeNumber,Attr.Age, Attr.DistanceFromHome,Attr.BusinessTravel, EmpExpRate.MonthlyIncome,
EmpExpRate.NumCompaniesWorked As 'PrevJobNo.', EmpExpRate.TotalWorkingYears as 'WorkEx' 
From Attr 
Join EmpExpRate on Attr.EmployeeNumber=EmpExpRate.EmployeeNumber
where EmpExpRate.MonthlyIncome=(Select MAX(MonthlyIncome) from EmpExpRate where EmpExpRate.Attrition=1);
/*Employee 787 was of age 55 years and he left the job to confirm this I will study his 
Job Satisfaction and Relationship Satisfaction*/
------------------------------------------------------------------------------------------

Select JobLevel,JobRole,JobSatisfaction,RelationshipSatisfaction,YearsWithCurrManager,YearsAtCompany 
from EmpWithMgr where EmployeeNumber=787;
/*The results confirm that he left because of his age. On this ground we can infer that other 
121 might have quit because of personal reasons. In the subsequent queries I will be testing this
inference in one by performing multiple joins*/
---------------------------------------------------------------------------------------------------

With Attr As (Select * from EmployeePersonalDetails 
where Attrition=1 and EnvironmentSatisfaction>=3)
Select Attr.EmployeeNumber,Attr.Age, Attr.DistanceFromHome,Attr.BusinessTravel,
EmpExpRate.MonthlyIncome,EmpExpRate.NumCompaniesWorked As 'PrevJobNo.', EmpExpRate.TotalWorkingYears as 'WorkEx',
EmpWithMgr.JobLevel, EmpWithMgr.JobRole, EmpWithMgr.JobSatisfaction,
EmpWithMgr.RelationshipSatisfaction as 'Relation'
From Attr 
Join EmpExpRate on Attr.EmployeeNumber=EmpExpRate.EmployeeNumber
Join EmpWithMgr on Attr.EmployeeNumber=EmpWithMgr.EmployeeNumber;
/*The above results have contradiction to previous inference as people were not happy with either job or relation.
This leads us to analyse further. We can however state that people with
high JobSatisfaction and RelationshipSatisfaction might have left because of job role, job level or monthly income.
In the next Query we are going to anlyse this inference.*/
--------------------------------------------------------------------------------------------------------------------

With Attr As (Select * from EmployeePersonalDetails 
where Attrition=1 and EnvironmentSatisfaction>=3)
Select Attr.EmployeeNumber,Attr.Age, Attr.DistanceFromHome,Attr.BusinessTravel,
EmpExpRate.MonthlyIncome,EmpExpRate.NumCompaniesWorked As 'PrevJobNo.', EmpExpRate.TotalWorkingYears as 'WorkEx',
EmpWithMgr.JobLevel,
EmpWithMgr.RelationshipSatisfaction as 'Relation'
From Attr 
Join EmpExpRate on Attr.EmployeeNumber=EmpExpRate.EmployeeNumber
Join EmpWithMgr on Attr.EmployeeNumber=EmpWithMgr.EmployeeNumber
where EmpWithMgr.JobSatisfaction>=3;
/*The results show that though they were satisfied with job many were unsatisfied with Relationships.
This means we will need to understand the relations with mananger. 68 people who left were satisfied with the job*/
-------------------------------------------------------------------------------------------------------------------
With Attr As (Select * from EmployeePersonalDetails 
where Attrition=1 and EnvironmentSatisfaction>=3)
Select Attr.EmployeeNumber,Attr.Age, Attr.DistanceFromHome,Attr.BusinessTravel,
EmpExpRate.MonthlyIncome,EmpExpRate.NumCompaniesWorked As 'PrevJobNo.', EmpExpRate.TotalWorkingYears as 'WorkEx',
EmpWithMgr.JobLevel, EmpWithMgr.YearsWithCurrManager
From Attr 
Join EmpExpRate on Attr.EmployeeNumber=EmpExpRate.EmployeeNumber
Join EmpWithMgr on Attr.EmployeeNumber=EmpWithMgr.EmployeeNumber
where EmpWithMgr.JobSatisfaction>=3 and EmpWithMgr.RelationshipSatisfaction<3;
/*Out of 68 26 people were unsatisfied with relations with staff.*/
/*The study so far indicates that there are multiple reasons for attrition. Therefore to make a better case study
I am switching to visual representations of data and if need be I will be finding relationships through
visualisations and use custom Sql Queries to enhance the inferrence. The study of possible outliars has broadened
the horizon of analysis and a comprehensive solution can be governing all domains*/
