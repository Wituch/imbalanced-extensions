using Microsoft.ML;

namespace ImbalancedExtensions
{
    public static class ImbalancedExtensions
    {
        public static int GetRowCount(this DataOperationsCatalog catalog, IDataView data)
        {
            return catalog.CreateEnumerable<object>(data, reuseRowObject: true).Count();
        }

        public static double GetBinaryClassRatio(this DataOperationsCatalog catalog, IDataView data, string labelColumnName = "Label")
        {
            var totalRowsCount = catalog.GetRowCount(data);
            var falseValuesCount = catalog.GetRowCount(catalog.FilterRowsByColumn(data, labelColumnName, lowerBound: 0, upperBound: 1));
            var trueValuesCount = totalRowsCount - falseValuesCount;
            var minorityClassCount = falseValuesCount > trueValuesCount ? trueValuesCount : falseValuesCount;
            return (double)minorityClassCount / totalRowsCount;
        }

        public static IDataView BinaryRandomUndersample<T>(this DataOperationsCatalog catalog, IDataView data, IEqualityComparer<T> equalityComparer, double ratio = 0.5, bool negativeValueIsMajority = true, string labelColumnName = "Label", int seed = 0) where T : class, new()
        {
            var random = new Random(seed);
            var elementsToDelete = new HashSet<T>(equalityComparer);
            var totalCount = catalog.GetRowCount(data);
            IDataView majorityClass = null;

            if (negativeValueIsMajority)
            {
                majorityClass = catalog.FilterRowsByColumn(data, labelColumnName, 0, 1);
            }
            else
            {
                majorityClass = catalog.FilterRowsByColumn(data, labelColumnName, 1, 1);
            }

            var majorityClassDataEnumerable = catalog.CreateEnumerable<T>(majorityClass, reuseRowObject: false).ToList();
            var majorityClassCount = majorityClassDataEnumerable.Count();
            double minorityClassCount = totalCount - majorityClassCount;
            var currentMinorityClassRatio = minorityClassCount / totalCount;

            while (currentMinorityClassRatio < ratio)
            {
                var randomIndex = random.Next(majorityClassCount);
                var elementToDelete = majorityClassDataEnumerable[randomIndex];

                if (!elementsToDelete.Contains(elementToDelete))
                {
                    elementsToDelete.Add(elementToDelete);
                    totalCount--;
                    currentMinorityClassRatio = minorityClassCount / totalCount;
                }
            }

            return catalog.FilterByCustomPredicate(data, (T row) => elementsToDelete.Contains(row));
        }

        public static IDataView BinaryRandomOversample<T>(this DataOperationsCatalog catalog, IDataView data, double ratio = 0.5, bool negativeValueIsMajority = true, string labelColumnName = "Label", int seed = 0) where T : class, new()
        {
            var random = new Random(seed);
            var elementsToAppend = new List<T>();
            var totalCount = catalog.GetRowCount(data);
            IDataView minorityClass = null;

            if (negativeValueIsMajority)
            {
                minorityClass = catalog.FilterRowsByColumn(data, labelColumnName, 1, 2);
            }
            else
            {
                minorityClass = catalog.FilterRowsByColumn(data, labelColumnName, 0, 1);
            }

            var minorityClassDataEnumerable = catalog.CreateEnumerable<T>(minorityClass, reuseRowObject: false).ToList();
            var originalMinorityClassCount = minorityClassDataEnumerable.Count();
            var currentMinorityClassCount = originalMinorityClassCount;
            var currentMinorityClassRatio = (double)originalMinorityClassCount / totalCount;

            while (currentMinorityClassRatio < ratio)
            {
                var randomIndex = random.Next(originalMinorityClassCount);
                var elementToDuplicate = minorityClassDataEnumerable[randomIndex];
                elementsToAppend.Add(elementToDuplicate);
                totalCount++;
                currentMinorityClassCount++;
                currentMinorityClassRatio = (double)currentMinorityClassCount / totalCount;
            }

            var finalData = catalog.CreateEnumerable<T>(data, reuseRowObject: false).ToList();
            finalData.AddRange(elementsToAppend);
            return catalog.LoadFromEnumerable(finalData);
        }
    }
}