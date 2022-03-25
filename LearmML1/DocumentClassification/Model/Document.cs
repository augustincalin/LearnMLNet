using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace DocumentClassification.Model
{
    internal class Document
    {
        [ColumnName(@"Id")]
        public float Id { get; set; }

        [ColumnName(@"Name")]
        public string Name { get; set; }

        [ColumnName(@"UseCase")]
        public string UseCase { get; set; }

        [ColumnName(@"BusinessCaseId")]
        public int BusinessCaseId { get; set; }

        [ColumnName(@"Designation")]
        public string Designation { get; set; }

        [ColumnName(@"Bank")]
        public string Bank { get; set; }

        [ColumnName(@"Currency")]
        public string Currency { get; set; }

        [ColumnName(@"Biller")]
        public string Biller { get; set; }

        [ColumnName(@"RawText")]
        public string RawText { get; set; }

    }
}
