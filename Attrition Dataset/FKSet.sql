use AcmeDataset;
CREATE PROCEDURE dynalter 
    @table NVARCHAR(128)
AS
BEGIN
    DECLARE @sql NVARCHAR(MAX);
    SET @sql = N'ALTER TABLE ' + QUOTENAME(@table) + 
               N' ADD FOREIGN KEY (EmployeeNumber) REFERENCES FullAttRep(EmployeeNumber)';
    EXEC sp_executesql @sql;
END;
Exec dynalter EmpEduJobDetails;
Exec dynalter EmpExpRate;
Exec dynalter EmpJobDetails;
Exec dynalter EmployeePersonalDetails;
Exec dynalter EmpPerformance;
Exec dynalter EmpWithMgr;
Exec dynalter EmpyearsSalAcme;