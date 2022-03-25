
using System.Data.SqlClient;
using DocumentClassification;
using DocumentClassification.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

string _modelPath = Path.Combine(_appPath, "model");
string _modelFileName = "model.zip";
Directory.CreateDirectory(_modelPath);

_modelPath = Path.Combine(_modelPath, _modelFileName);


string _connectionString = "Data Source=.;Initial Catalog=COPDB;Integrated Security=True";

MLContext _mlContext;
PredictionEngine<Document, DocumentClassificationOutput> _predictionEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;

_mlContext = new MLContext();

DatabaseLoader databaseLoader = _mlContext.Data.CreateDatabaseLoader<Document>();
string sqlCommandTrain = "SELECT * FROM Documents WHERE IsTest = 0";
// string sqlCommandTrain = "SELECT * FROM Documents";
string sqlCommandTest = "SELECT * FROM Documents WHERE IsTest = 1";



DatabaseSource dbSourceTrain = new DatabaseSource(SqlClientFactory.Instance, _connectionString, sqlCommandTrain);
DatabaseSource dbSourceTest = new DatabaseSource(SqlClientFactory.Instance, _connectionString, sqlCommandTest);

_trainingDataView = databaseLoader.Load(dbSourceTrain);

var pipeline = ProcessData();
var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

Evaluate(_trainingDataView.Schema);

PredictIssue();

IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"RawText", outputColumnName: @"RawText")
                        .Append(_mlContext.Transforms.Concatenate(@"Features", new[] { @"RawText" }))
                        .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"UseCase", inputColumnName: @"UseCase"))
                        .Append(_mlContext.Transforms.NormalizeMinMax(@"Features", @"Features"));
    return pipeline;
}

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
            new LbfgsMaximumEntropyMulticlassTrainer.Options()
            {
                L1Regularization = 1F,
                L2Regularization = 1F,
                LabelColumnName = @"UseCase",
                FeatureColumnName = @"Features"
            }
            ))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

    _trainedModel = trainingPipeline.Fit(trainingDataView);

    _predictionEngine = _mlContext.Model.CreatePredictionEngine<Document, DocumentClassificationOutput>(_trainedModel);

    Document document = new()
    {
        Id = 999,
        Bank = "A Bank",
        Biller = "A Biller",
        BusinessCaseId = -1,
        Currency = "RR",
        Designation = "UNK",
        Name = "A Name",
        UseCase = "A UseCase",
        RawText = "Some raw text"
    };
    var prediction = _predictionEngine.Predict(document);
    Console.WriteLine($"Predicted Category: {prediction.PredictedLabel}");
    return trainingPipeline;
}

void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = databaseLoader.Load(dbSourceTest);

    var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView), labelColumnName: "UseCase");

    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");

    EvaluateAdditionalMetrics(testMetrics.ConfusionMatrix);

    SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
}

void EvaluateAdditionalMetrics(ConfusionMatrix confusionMatrix)
{
    var measures = confusionMatrix.GetAllMeasures();
    Console.WriteLine($"*       Additional metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       Macro.F1Score:                  {measures.Macro.F1Score:#.###}");
    Console.WriteLine($"*       Macro.FBetaMeasurePrecision:    {measures.Macro.FBetaMeasurePrecision:#.###}");
    Console.WriteLine($"*       Macro.FBetaMeasureRecall:       {measures.Macro.FBetaMeasureRecall:#.###}");
    Console.WriteLine($"*       Macro.Precision:                {measures.Macro.Precision:#.###}");
    Console.WriteLine($"*       Macro.Recall:                   {measures.Macro.Recall:#.###}");

    Console.WriteLine($"*       Micro.F1Score:                  {measures.Micro.F1Score:#.###}");
    Console.WriteLine($"*       Micro.FBetaMeasurePrecision:    {measures.Micro.FBetaMeasurePrecision:#.###}");
    Console.WriteLine($"*       Micro.FBetaMeasureRecall:       {measures.Micro.FBetaMeasureRecall:#.###}");
    Console.WriteLine($"*       Micro.Precision:                {measures.Micro.Precision:#.###}");
    Console.WriteLine($"*       Micro.Recall:                   {measures.Micro.Recall:#.###}");

    Console.WriteLine($"*       Weighted.F1Score:               {measures.Weighted.F1Score:#.###}");
    Console.WriteLine($"*       Weighted.FBetaMeasurePrecision: {measures.Weighted.FBetaMeasurePrecision:#.###}");
    Console.WriteLine($"*       Weighted.FBetaMeasureRecall:    {measures.Weighted.FBetaMeasureRecall:#.###}");
    Console.WriteLine($"*       Weighted.Precision:             {measures.Weighted.Precision:#.###}");
    Console.WriteLine($"*       Weighted.Recall:                {measures.Weighted.Recall:#.###}");

    Console.WriteLine($"*       Specifity:                      {measures.Specificity:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
}

void PredictIssue()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

    Document document = new()
    {
        Id = 999,
        Bank = "A Bank",
        Biller = "A Biller",
        BusinessCaseId = -1,
        Currency = "RR",
        Designation = "UNK",
        Name = "A Name",
        UseCase = "A UseCase",
        RawText = "Some raw text"
    };

    _predictionEngine = _mlContext.Model.CreatePredictionEngine<Document, DocumentClassificationOutput>(loadedModel);
    var prediction = _predictionEngine.Predict(document);

    Console.WriteLine($"Predicted: {prediction.PredictedLabel}");
}

void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
}