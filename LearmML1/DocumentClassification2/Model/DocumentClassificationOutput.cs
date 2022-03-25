using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace DocumentClassification2.Model
{
    internal class DocumentClassificationOutput
    {
        [ColumnName(@"Id")]
        public float Id { get; set; }

        [ColumnName(@"Name")]
        public string Name { get; set; }

        [ColumnName(@"UseCase")]
        public float[] UseCase { get; set; }

        [ColumnName(@"BusinessCaseId")]
        public float BusinessCaseId { get; set; }

        [ColumnName(@"Designation")]
        public string Designation { get; set; }

        [ColumnName(@"Bank")]
        public string Bank { get; set; }

        [ColumnName(@"Currency")]
        public string Currency { get; set; }

        [ColumnName(@"Biller")]
        public string Biller { get; set; }

        [ColumnName(@"RawText")]
        public float[] RawText { get; set; }

        [ColumnName(@"IsTest")]
        public bool IsTest { get; set; }

        [ColumnName(@"Category")]
        public uint Category { get; set; }

        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }
}
