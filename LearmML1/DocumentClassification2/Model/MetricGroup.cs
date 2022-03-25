using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace DocumentClassification2.Model
{
    internal class MetricGroup
    {
        public List<QualityMeasures> QualityMeasureList { get; set; }
        public MulticlassClassificationMetrics ValidationMetrics { get; set; }

        public QualityMeasures QualityMeasures { get; set; }

    }
}
