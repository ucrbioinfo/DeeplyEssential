# author: Abid Hasan
# University of california Riverside

# importing packages
import sys
import time

# import from other class
from AnalyzeParameter import *
from FileProcessing import *
from FeatureGeneration import *
from DatasetProcessing import *
from DNNModel import *

# in tool parameter
paramDict = {
    'bins': 1,
    'trainingProp': 0.8,
    'repeat': 10
}


def main():
    essential_dir_path = sys.argv[
        1]  # path to essential gene directory, directory contains fasta read, annotation read, portein seq read, mcl read
    non_essential_dir_path = sys.argv[
        2]  # path to non_essential gene directory, directory contains fasta read, annotation read, portein seq read, mcl read
    mcl_file_path = sys.argv[3]  # combined MCL file path
    dataset = sys.argv[4]  # dataset information
    option = sys.argv[5]  # -c complete data, -gp for gram positive, -gn for gram negative
    experimentName = sys.argv[6]  # name of the experiment

    # program start time
    start_time = time.time()

    # processing parameters
    param = ParameterProcessing(essential_dir_path, non_essential_dir_path, dataset, mcl_file_path)

    # read the files and extract information from the file
    read = ReadFiles(param)
    read.getDatasetFileInfo()  # read the dataset for other method to access dataset info

    # process the files and generate features from sequence and other files
    feat = FeatureProcessing(read)
    feat.getFeatures(paramDict['bins'])

    # get features and build training/testing dataset
    data = ProcessData(read, feat, paramDict['trainingProp'], option)

    # creating file to store evaluation statistics
    fWrite = open(experimentName + '.tab', 'w')
    fWrite.write("Experiment Name: " + str(experimentName) + '\n')
    fWrite.write("Number of training samples: " + str(data.getTrainingData().shape[0]) + '\n')
    fWrite.write("Number of validation samples: " + str(data.getValidationData().shape[0]) + '\n')
    fWrite.write("Number of testing samples: " + str(data.getTestingData().shape[0]) + '\n')
    fWrite.write("Number of features: " + str(data.getTrainingData().shape[1]) + '\n')
    fWrite.write("Iteration" + "\t" + "ROC_AUC" + "\t" + "Avg. Precision" + "\t" +
                 "Sensitivity" + "\t" + "Specificity" + "\t" + "PPV" + "\t" + "Accuracy" + "\n")

    # dict to store evaluation statistics to calculate average values
    evaluationValueForAvg = {
        'roc_auc': 0.,
        'precision': 0.,
        'sensitivity': 0.,
        'specificity': 0.,
        'PPV': 0.,
        'accuracy': 0.
    }

    # build DNN model
    if os.path.exists(experimentName + 'True_positives.txt'):
        os.remove(experimentName + 'True_positives.txt')
    if os.path.exists(experimentName + 'False_positives.txt'):
        os.remove(experimentName + 'False_positives.txt')
    if os.path.exists(experimentName + 'Thresholds.txt'):
        os.remove(experimentName + 'Thresholds.txt')

    f_tp = open(experimentName + 'True_positives.txt', 'a')
    f_fp = open(experimentName + 'False_positives.txt', 'a')
    f_th = open(experimentName + 'Thresholds.txt', 'a')

    for i in range(0, paramDict['repeat']):
        model = BuildDNNModel(data, paramDict['bins'], f_tp, f_fp, f_th)
        evaluationDict = model.getEvaluationStat()

        print evaluationDict

        writeEvaluationStat(evaluationDict, fWrite, i + 1)

        evaluationValueForAvg['roc_auc'] += evaluationDict['roc_auc']
        evaluationValueForAvg['precision'] += evaluationDict['precision']
        evaluationValueForAvg['sensitivity'] += evaluationDict['sensitivity']
        evaluationValueForAvg['specificity'] += evaluationDict['specificity']
        evaluationValueForAvg['PPV'] += evaluationDict['PPV']
        evaluationValueForAvg['accuracy'] += evaluationDict['accuracy']

    for value in evaluationValueForAvg:
        evaluationValueForAvg[value] = float(evaluationValueForAvg[value]) / paramDict['repeat']

    writeEvaluationStat(evaluationValueForAvg, fWrite, 'Avg.')
    end_time = time.time()
    fWrite.write("Execution time: " + str(end_time - start_time) + " sec.")
    fWrite.close()

    f_tp.close()
    f_fp.close()
    f_th.close()


# writes the evaluation statistics
def writeEvaluationStat(evaluationDict, fWrite, iteration):
    fWrite.write(str(iteration) + "\t" + str(evaluationDict['roc_auc']) + "\t" +
                 str(evaluationDict['precision']) + '\t' + str(evaluationDict['sensitivity']) + '\t' +
                 str(evaluationDict['specificity']) + '\t' + str(evaluationDict['PPV']) + '\t' +
                 str(evaluationDict['accuracy']) + '\n')


if __name__ == "__main__":
    main()
