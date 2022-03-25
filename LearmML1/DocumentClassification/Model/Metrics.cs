using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DocumentClassification.Model
{
    internal class Metrics
    {
        public double F1Score { get; set; }
        public double FBetaMeasurePrecision { get; set; }
        public double FBetaMeasureRecall { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
    }
}
