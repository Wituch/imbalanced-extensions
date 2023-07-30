using ImbalancedExtensions.Sample;
using Microsoft.ML;
using ImbalancedExtensions;

//Create a new context for ML.NET operations
var mlContext = new MLContext();

//Create a list of data samples
var dataList = new List<SampleData>
{
    new SampleData { X = 1, Y = 1, Z = 0 },
    new SampleData { X = 2, Y = 2, Z = 1 },
    new SampleData { X = 3, Y = 3, Z = 0 },
    new SampleData { X = 4, Y = 4, Z = 0 },
    new SampleData { X = 5, Y = 5, Z = 0 },
    new SampleData { X = 6, Y = 6, Z = 0 },
    new SampleData { X = 7, Y = 7, Z = 0 },
    new SampleData { X = 8, Y = 8, Z = 1 },
    new SampleData { X = 9, Y = 9, Z = 0 },
    new SampleData { X = 0, Y = 0, Z = 0 },
};

//Load data list into IDataView, used by ML.NET
var data = mlContext.Data.LoadFromEnumerable(dataList);

//Perform random oversampling on data - duplicate random minority class rows until desired ratio is met
var oversampledData = mlContext.Data.BinaryRandomOversample<SampleData>(data);

//Perform random undersampling on data - delete random majority class rows until desired ratio is met
var undersampledData = mlContext.Data.BinaryRandomUndersample(data, new SampleDataEqualityComparer());

//Display experiment results on console
Console.WriteLine("Initial data set:");
foreach(var item in dataList)
{
    Console.WriteLine($"X: {item.X} Y: {item.Y} Z: {item.Z}");
}

Console.WriteLine("\nData set after random oversampling:");
foreach (var item in mlContext.Data.CreateEnumerable<SampleData>(oversampledData, false))
{
    Console.WriteLine($"X: {item.X} Y: {item.Y} Z: {item.Z}");
}

Console.WriteLine("\nData set after random undersampling:");
foreach (var item in mlContext.Data.CreateEnumerable<SampleData>(undersampledData, false))
{
    Console.WriteLine($"X: {item.X} Y: {item.Y} Z: {item.Z}");
}

