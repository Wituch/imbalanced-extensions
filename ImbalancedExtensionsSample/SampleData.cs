using Microsoft.ML.Data;
using System.Diagnostics.CodeAnalysis;

namespace ImbalancedExtensions.Sample
{
    public class SampleData
    {
        public double X { get; set; }
        public double Y { get; set; }
        [ColumnName("Label")]
        public double Z { get; set; }
    }

    public class SampleDataEqualityComparer : IEqualityComparer<SampleData>
    {
        public bool Equals(SampleData? x, SampleData? y)
        {
            if (x is null || y is null)
            {
                return false;
            }
            
            return x.X == y.X && x.Y == y.Y && x.Z == y.Z;
        }

        public int GetHashCode([DisallowNull] SampleData obj)
        {
            return obj.X.GetHashCode() ^ obj.Y.GetHashCode() ^ obj.Z.GetHashCode();
        }
    }
}
