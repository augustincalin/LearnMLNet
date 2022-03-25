using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace DocumentClassification.Model
{
    internal class MetricGroup
    {
        public List<QualityMeasures> QualityMeasureList { get; set; }
        public MulticlassClassificationMetrics ValidationMetrics { get; set; }

        public QualityMeasures QualityMeasures { get; set; }

    }
}
