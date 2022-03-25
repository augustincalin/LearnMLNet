using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DocumentClassification.Model
{
    internal class QualityMeasures
    {
        public QualityMeasures()
        {
            Weighted = new Metrics();
            Macro = new Metrics();
            Micro = new Metrics();
        }

        public Metrics Weighted { get; set; }
        public Metrics Macro { get; set; }
        public Metrics Micro { get; set; }
        public double Specificity { get; set; }
    }
}
