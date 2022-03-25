using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DocumentClassification2.Model;
using Microsoft.ML.Data;

namespace DocumentClassification2
{
    internal static class ConfusionMatrixExtension
    {
        internal static QualityMeasures GetAllMeasures(this ConfusionMatrix confusionMatrix)
        {
            QualityMeasures qualityMeasures = new QualityMeasures();

            qualityMeasures.Weighted.F1Score = GetWeightedF1Score(confusionMatrix);
            qualityMeasures.Weighted.FBetaMeasurePrecision = GetWeightedFBetaMeasure(confusionMatrix, BetaMeasures.TowardsPrecision);
            qualityMeasures.Weighted.FBetaMeasureRecall = GetWeightedFBetaMeasure(confusionMatrix, BetaMeasures.TowardsRecall);
            qualityMeasures.Weighted.Precision = GetWeightedPrecision(confusionMatrix);
            qualityMeasures.Weighted.Recall = GetWeightedRecall(confusionMatrix);

            qualityMeasures.Micro.F1Score = GetMicroF1Score(confusionMatrix);
            qualityMeasures.Micro.FBetaMeasurePrecision = GetMicroFBetaMeasure(confusionMatrix);
            qualityMeasures.Micro.FBetaMeasureRecall = GetMicroFBetaMeasure(confusionMatrix);
            qualityMeasures.Micro.Precision = GetMicroPrecision(confusionMatrix);
            qualityMeasures.Micro.Recall = GetMicroRecall(confusionMatrix);

            qualityMeasures.Macro.F1Score = GetMacroF1Score(confusionMatrix);
            qualityMeasures.Macro.FBetaMeasurePrecision = GetMacroFBetaMeasure(confusionMatrix, BetaMeasures.TowardsPrecision);
            qualityMeasures.Macro.FBetaMeasureRecall = GetMacroFBetaMeasure(confusionMatrix, BetaMeasures.TowardsRecall);
            qualityMeasures.Macro.Precision = GetMacroPrecision(confusionMatrix);
            qualityMeasures.Macro.Recall = GetMacroRecall(confusionMatrix);

            qualityMeasures.Specificity = GetSpecificity(confusionMatrix);

            return qualityMeasures;
        }
        public static double GetWeightedF1Score(this ConfusionMatrix confusionMatrix, double betaFactor = 1)
        {
            IReadOnlyList<double> perClassPrecision = confusionMatrix.PerClassPrecision;
            IReadOnlyList<double> perClassRecall = confusionMatrix.PerClassRecall;
            int numberOfClasses = confusionMatrix.NumberOfClasses;
            double weightedf1score = 0;

            IReadOnlyList<IReadOnlyList<double>> counts = confusionMatrix.Counts;

            double[] sumOfEachClass = GetSumOfEachClass(counts, numberOfClasses);

            for (int i = 0; i < numberOfClasses; i++)
            {
                double f1score = GetF1Score(perClassPrecision[i], perClassRecall[i], betaFactor);
                weightedf1score += f1score * sumOfEachClass[i];

            }

            double totalSamples = sumOfEachClass.Sum();
            return weightedf1score / totalSamples;
        }

        //
        // Summary:
        //     It is the general form of F measure — Beta 0.5 & 2 are usually used as
        //     measures, 0.5 indicates the Inclination towards Precision whereas 2 favors
        //     Recall giving it twice the weightage compared to precision.
        public static double GetWeightedFBetaMeasure(this ConfusionMatrix confusionMatrix, BetaMeasures betaMeasure)
        {
            double betaFactor = GetBetaFactor(betaMeasure);
            return GetWeightedF1Score(confusionMatrix, betaFactor);
        }

        public static double GetWeightedRecall(this ConfusionMatrix confusionMatrix)
        {
            IReadOnlyList<double> perClassRecall = confusionMatrix.PerClassRecall;
            int numberOfClasses = confusionMatrix.NumberOfClasses;
            double weightedRecallscore = 0;
            IReadOnlyList<IReadOnlyList<double>> counts = confusionMatrix.Counts;
            double[] sumOfEachClass = GetSumOfEachClass(counts, numberOfClasses);

            for (int i = 0; i < numberOfClasses; i++)
            {
                weightedRecallscore += perClassRecall[i] * sumOfEachClass[i];
            }

            double totalSamples = sumOfEachClass.Sum();
            return weightedRecallscore / totalSamples;
        }
        public static double GetWeightedPrecision(this ConfusionMatrix confusionMatrix)
        {
            IReadOnlyList<double> perClassPrecision = confusionMatrix.PerClassPrecision;
            int numberOfClasses = confusionMatrix.NumberOfClasses;
            double weightedPrecisionscore = 0;
            IReadOnlyList<IReadOnlyList<double>> counts = confusionMatrix.Counts;
            double[] sumOfEachClass = GetSumOfEachClass(counts, numberOfClasses);

            for (int i = 0; i < numberOfClasses; i++)
            {
                weightedPrecisionscore += perClassPrecision[i] * sumOfEachClass[i];
            }

            double totalSamples = sumOfEachClass.Sum();
            return weightedPrecisionscore / totalSamples;
        }


        public static double GetMacroF1Score(this ConfusionMatrix confusionMatrix, double betaFactor = 1)
        {
            IReadOnlyList<double> perClassPrecision = confusionMatrix.PerClassPrecision;
            IReadOnlyList<double> perClassRecall = confusionMatrix.PerClassRecall;
            int numberOfClasses = confusionMatrix.NumberOfClasses;
            double macrof1score = 0;

            for (int i = 0; i < numberOfClasses; i++)
            {
                double f1score = GetF1Score(perClassPrecision[i], perClassRecall[i], betaFactor);
                macrof1score += f1score;
            }

            macrof1score /= numberOfClasses;
            return macrof1score;
        }

        //
        // Summary:
        //     It is the general form of F measure — Beta 0.5 & 2 are usually used as
        //     measures, 0.5 indicates the Inclination towards Precision whereas 2 favors
        //     Recall giving it twice the weightage compared to precision.
        public static double GetMacroFBetaMeasure(this ConfusionMatrix confusionMatrix, BetaMeasures betaMeasure)
        {
            double betaFactor = GetBetaFactor(betaMeasure);
            return GetMacroF1Score(confusionMatrix, betaFactor);
        }

        public static double GetMacroRecall(this ConfusionMatrix confusionMatrix)
        {
            IReadOnlyList<double> perClassRecall = confusionMatrix.PerClassRecall;
            double macroRecall = perClassRecall.Average();
            return macroRecall;
        }

        public static double GetMacroPrecision(this ConfusionMatrix confusionMatrix)
        {
            IReadOnlyList<double> perClassPrecision = confusionMatrix.PerClassPrecision;
            double macroPrecision = perClassPrecision.Average();
            return macroPrecision;
        }

        public static double GetMicroF1Score(this ConfusionMatrix confusionMatrix)
        {
            IReadOnlyList<IReadOnlyList<double>> counts = confusionMatrix.Counts;
            int numberOfClasses = confusionMatrix.NumberOfClasses;
            double sumOfTruePositives = 0;
            double sumOfFalsePositives = 0;

            for (int i = 0; i < numberOfClasses; i++)
            {
                for (int j = 0; j < numberOfClasses; j++)
                {
                    if (i == j)
                    {
                        sumOfTruePositives += counts[i][j];
                    }
                    else
                    {
                        sumOfFalsePositives += counts[i][j];
                    }
                }
            }

            double microf1Score = sumOfTruePositives / (sumOfFalsePositives + sumOfTruePositives);
            return microf1Score;
        }

        //
        // Summary:
        //     It is the general form of F measure — Beta 0.5 & 2 are usually used as
        //     measures, 0.5 indicates the Inclination towards Precision whereas 2 favors
        //     Recall giving it twice the weightage compared to precision.
        public static double GetMicroFBetaMeasure(this ConfusionMatrix confusionMatrix)
        {
            return GetMicroF1Score(confusionMatrix);
        }

        public static double GetMicroRecall(this ConfusionMatrix confusionMatrix)
        {
            return GetMicroF1Score(confusionMatrix);
        }

        public static double GetMicroPrecision(this ConfusionMatrix confusionMatrix)
        {
            return GetMicroF1Score(confusionMatrix);
        }

        //
        // Summary:
        //    It is also referred to as ‘True Negative Rate’ (Proportion of actual negatives
        //    that are correctly identified), i.e. more True Negatives the data hold the
        //    higher its Specificity.
        public static double GetSpecificity(this ConfusionMatrix confusionMatrix)
        {
            //Todo: To be implemented
            return 0;
        }

        private static double GetF1Score(double precision, double recall, double beta = 1)
        {
            double f1score = (1 + beta * beta) * (precision * recall) / (((beta * beta) * precision) + recall);
            if (precision == 0 && recall == 0)
            {
                f1score = 0;
            }
            return f1score;

        }

        private static double[] GetSumOfEachClass(IReadOnlyList<IReadOnlyList<double>> counts, int numberOfClasses)
        {
            double[] sumOfEachClass = new double[numberOfClasses];

            for (int i = 0; i < numberOfClasses; i++)
            {
                for (int j = 0; j < numberOfClasses; j++)
                {
                    sumOfEachClass[i] = sumOfEachClass[i] + counts[j][i];
                }
            }

            return sumOfEachClass;
        }

        private static double GetBetaFactor(BetaMeasures betaMeasure)
        {
            var betaFactor = betaMeasure switch
            {
                BetaMeasures.TowardsPrecision => 0.5,
                BetaMeasures.TowardsRecall => 2,
                _ => 1,
            };
            return betaFactor;
        }
    }
}
