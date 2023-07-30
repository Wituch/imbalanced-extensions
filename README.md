# Imbalanced Extensions
Extension methods for ML.NET allowing imbalanced data sampling, inspired by imbalanced-learn library for Python: https://imbalanced-learn.org/

# Motivation
ML.NET framework currently does not support operations to deal with imbalanced data set, such as undersampling or oversampling. The lack of such functionalities has been already spotted and reported: https://github.com/dotnet/machinelearning/issues/6523, but so far there is no timeline when this could possibly be delivered. The goal of this project is to provide temporary solution to handle imbalanced data scenarios.

# Functionalities
## Binary random oversampling
Performs random oversampling on provided IDataView - data rows from minority class are picked at random, duplicated and appended to dataset, unit desired binary class ratio (default  is 0,5) is met.
```
//Perform random oversampling on data - duplicate random minority class rows until desired ratio is met
var oversampledData = mlContext.Data.BinaryRandomOversample<SampleData>(data);
```
## Binary random undersampling
Performs random undersampling on provided IDataView - data rows from majority class are picked at random and removed from dataset, unit desired binary class ratio (default  is 0,5) is met. In current version, it is required to implement EqualityComparer for used data object class in order to use this method.
```
//Perform random undersampling on data - delete random majority class rows until desired ratio is met
var undersampledData = mlContext.Data.BinaryRandomUndersample(data, new SampleDataEqualityComparer());
```
## Get row count
Returns number of data rows in provided IDataView. This method has been created because of existing IDataView.GetRowCount() method not being able to always return row count (see: https://github.com/dotnet/machinelearning/issues/3450).
```
var rowCount = mlContext.Data.GetRowCount(data);
```
## Get binary class ratio
Returns ratio of minority class records to majority class records in provided IDataView.
```
var binaryClassRatio = mlContext.Data.GetBinaryClassRatio(data);
```

# Limitations
Currently oversampling and undersampling functionalities are only limited to binary classification scenarios.
