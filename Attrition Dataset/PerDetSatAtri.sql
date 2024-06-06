/*There are multiple factors that will affect attrition therefore, in this section of queries,
we are going to study attrition of people who left based on differet tables. We can then use these
queries in tableau. If need be we will create custom queries to study multiple variables together*/
--------------------------------------------------------------------------------------------------
Use AcmeDataset;
Select * from EmployeePersonalDetails where Attrition=1;/*Attrition=1 tells that employess have quit*/
/*The above query gives us an unepected result that people fairly satisfied with environment
(EnvironmentSatisfaction>=3) have quit too. To understand this we will be discussing this further and 
therefore adress multiple problems from different sets of tables.*/
------------------------------------------------------------------------------------------------------

Select * from EmployeePersonalDetails where Attrition=1 and EnvironmentSatisfaction>=3;
/*The result shows that 122 people fall under this category and is not dependent on distance.
To study this purpose I intend to study and compare this result with other tables. */
----------------------------------------------------------------------------------------------
Select COUNT(Gender) As TotalFemaleCount from EmployeePersonalDetails where Gender='Female';
/*results show that a total of 588 female employees ae associated with ACME out of 1470 total employees*/
----------------------------------------------------------------------------------------------------------

Select * from EmployeePersonalDetails where Attrition=1 and EnvironmentSatisfaction>=3 and Gender='Female';
/*results show that a total of 39 women out of 588 have quit.*/
-----------------------------------------------------------------------------------------------------------

Select Count(MaritalStatus) as MarriedWomen from EmployeePersonalDetails
where MaritalStatus='Married' and Gender='Female';
/*A total of 272 women are married*/
-------------------------------------

Select * from EmployeePersonalDetails
where Attrition=1 and EnvironmentSatisfaction>=3 and Gender='Female' and MaritalStatus='Married';
/*The above results imply that only 16 out of 39 were married.Also this indicates that only 16 of 272 married women
quit, implying theier primary reasons might be different from the above queries*/
--------------------------------------------------------------------------------------------------------------------

/*Having such a general idea we can now move forward to what was bothering those who rated high yet resigned*/
/*The next step would be to study attrition in those who have given low ratings*/