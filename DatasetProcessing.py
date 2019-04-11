import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# returns the list of genes to be used
def getGeneDict(accessionList, annotationDict):
    geneNameDict = dict()

    for geneName in annotationDict:
        if annotationDict[geneName][2] in accessionList:
            geneNameDict[geneName] = True

    return geneNameDict


# returns MCL based essential and non essential gene list
def getGenesFromMCLStat(MCLStatDict, combinedMCLDict, essentialGeneNameDict, nonEssentialGeneNameDict, lower, upper):
    mclEssentialGeneDict = dict()
    mclNonEssentialGeneDict = dict()

    for cluster in MCLStatDict:
        if lower <= cluster <= upper:
            geneList = combinedMCLDict[cluster]
            for i in range(0, len(geneList)):
                if 'DEG' in geneList[i]:
                    if geneList[i] in essentialGeneNameDict:
                        mclEssentialGeneDict[geneList[i]] = True
                elif 'DNEG' in geneList[i]:
                    if geneList[i] in nonEssentialGeneNameDict:
                        mclNonEssentialGeneDict[geneList[i]] = True

    return mclEssentialGeneDict, mclNonEssentialGeneDict


# method to split dataset
def combineAndSplitData(EssentialGeneFeatTable, NonEssentialGeneFeatTable, trainingProp, f_e, f_ne):
    # get the gene length of essential/non-essential genes
    essential_gene_length_list = np.array(EssentialGeneFeatTable)[:, 0]
    non_essential_gene_length_list = np.array(NonEssentialGeneFeatTable)[:, 0]

    for i in range(len(essential_gene_length_list)):
        f_e.write(str(essential_gene_length_list[i]) + '\n')

    for i in range(len(non_essential_gene_length_list)):
        f_ne.write(str(non_essential_gene_length_list[i]) + '\n')

    completeData = np.vstack((np.array(EssentialGeneFeatTable), np.array(NonEssentialGeneFeatTable)))
    completeResizedData = resizeData(EssentialGeneFeatTable, NonEssentialGeneFeatTable)

    # calculate the correlation matrix
    completeResizedData_df = pd.DataFrame(completeResizedData)

    # for feature correlation analysis
    corr = completeResizedData_df.corr()
    #
    # print min((corr.min()).tolist())    # -0.5817
    #
    corr_feat_list = list()

    for i in range(corr.shape[0]):
        for j in range(i, corr.shape[1]):
            if (corr[i][j] >= 0.90 or corr[i][j] <= -0.90) and i != j:
                corr_feat_list.append([i, j])

    # calculating training, validation, testing data portion
    validationProp, testingProp = float(1 - trainingProp) / 2, float(1 - trainingProp) / 2

    # shuffling the data to mix the data before splitting the dataset into training, validation and testing data
    np.random.shuffle(completeResizedData)

    # getting the shape of the reSized dataset to find the training, validation and testing size
    row, col = completeResizedData.shape
    trainingSize = int(row * trainingProp)
    validationSize = int(row * validationProp)
    testingSize = int(row * testingProp)

    trainingData = completeResizedData[:trainingSize, :]
    validationData = completeResizedData[trainingSize:(trainingSize + validationSize), :]
    testingData = completeResizedData[(trainingSize + validationSize):, :]

    return trainingData, validationData, testingData, corr_feat_list


class ProcessData(object):

    def __init__(self, read, feat, trainingProp, option, ExpName):
        super(ProcessData, self).__init__()
        self.feat = feat
        self.ExpName = ExpName

        # define which set of genes to work with
        self.CompleteDataAccession = read.getCompleteListOrganismAccession()
        self.GramPositveDataAccession = read.getGramPositiveOrganismAccession()
        self.GramNegativeDataAccession = read.getGramNegativeOrganismAccession()

        self.EssentialGeneSeqInfo = read.getEssentialGeneSeqInfo()
        self.EssentialProteinSeqInfo = read.getEssentialProteinInfo()
        self.EssentialAnnotationInfo = read.getEssentialGeneAnnoInfo()

        self.NonEssentialGeneSeqInfo = read.getNonEssentialGeneSeqInfo()
        self.NonEssentialProteinSeqInfo = read.getNonEssentialProteinInfo()
        self.NonEssentialAnnotationInfo = read.getNonEssentialGeneAnnoInfo()

        self.MCLStatDict = feat.getMCLStatDict()
        self.combinedMCLDict = feat.getCombinedMCLDict()

        # build data set for complete, gram positive and gram negative data
        self.completeEssentialGeneNameDict = getGeneDict(self.CompleteDataAccession, self.EssentialAnnotationInfo)
        self.gramPositiveEssentialGeneNameDict = getGeneDict(self.GramPositveDataAccession,
                                                             self.EssentialAnnotationInfo)
        self.gramNegativeEssentialGeneNameDict = getGeneDict(self.GramNegativeDataAccession,
                                                             self.EssentialAnnotationInfo)

        self.completeNonEssentialGeneNameDict = getGeneDict(self.CompleteDataAccession, self.NonEssentialAnnotationInfo)
        self.gramPositiveNonEssentialGeneNameDict = getGeneDict(self.GramPositveDataAccession,
                                                                self.NonEssentialAnnotationInfo)
        self.gramNegativeNonEssentialGeneNameDict = getGeneDict(self.GramNegativeDataAccession,
                                                                self.NonEssentialAnnotationInfo)

        # Building training, validation and testing data from the mcl gene clusters. This data is build by combining
        # similar genes so a higher accuracy is expected.
        self.TrainMCLEssentialGeneNameDict, self.TrainMCLNonEssentialGeneNameDict = getGenesFromMCLStat(
            self.MCLStatDict, self.combinedMCLDict, self.completeEssentialGeneNameDict, self.completeNonEssentialGeneNameDict, 0, 500)
        self.ValidMCLEssentialGeneNameDict, self.ValidMCLNonEssentialGeneNameDict = getGenesFromMCLStat(
            self.MCLStatDict, self.combinedMCLDict, self.completeEssentialGeneNameDict, self.completeNonEssentialGeneNameDict, 501, 650)
        self.TestMCLEssentialGeneNameDict, self.TestMCLNonEssentialGeneNameDict = getGenesFromMCLStat(self.MCLStatDict,
                                                                                                      self.combinedMCLDict,
                                                                                                      self.completeEssentialGeneNameDict, self.completeNonEssentialGeneNameDict,
                                                                                                      651, 1200)
        # building feature table and training testing dataset
        if option == '-c':
            self.EssentialGeneFeatTable = getGeneFeatTable(feat, self.completeEssentialGeneNameDict, classLabel=1)
            self.NonEssentialGeneFeatTable = getGeneFeatTable(feat, self.completeNonEssentialGeneNameDict, classLabel=0)

            f_essential_gene_length = open(str(self.ExpName) +  '_Essential_gene_length.txt', 'w')
            f_non_essential_gene_length = open(str(self.ExpName) +  '_Non_essential_gene_length.txt', 'w')

            self.trainingData, self.validationData, self.testingData, self.corr_feats = combineAndSplitData(self.EssentialGeneFeatTable,
                                                                                           self.NonEssentialGeneFeatTable,
                                                                                           trainingProp,
                                                                                           f_essential_gene_length,
                                                                                           f_non_essential_gene_length)
            f_essential_gene_length.close()
            f_non_essential_gene_length.close()

        elif option == '-gp':
            self.EssentialGeneFeatTable = getGeneFeatTable(feat, self.gramPositiveEssentialGeneNameDict, classLabel=1)
            self.NonEssentialGeneFeatTable = getGeneFeatTable(feat, self.gramPositiveNonEssentialGeneNameDict,
                                                              classLabel=0)

            f_gp_essential_gene_length = open(str(self.ExpName) +  '_GP_essential_gene_length.txt', 'w')
            f_gp_non_essential_gene_length = open(str(self.ExpName) +  '_GP_non_essential_gene_length.txt', 'w')

            self.trainingData, self.validationData, self.testingData, _ = combineAndSplitData(self.EssentialGeneFeatTable,
                                                                                           self.NonEssentialGeneFeatTable,
                                                                                           trainingProp,
                                                                                           f_gp_essential_gene_length,
                                                                                           f_gp_non_essential_gene_length)
            f_gp_essential_gene_length.close()
            f_gp_non_essential_gene_length.close()

        elif option == '-gn':
            self.EssentialGeneFeatTable = getGeneFeatTable(feat, self.gramNegativeEssentialGeneNameDict, classLabel=1)
            self.NonEssentialGeneFeatTable = getGeneFeatTable(feat, self.gramNegativeNonEssentialGeneNameDict,
                                                              classLabel=0)

            f_gn_essential_gene_length = open(str(self.ExpName) +  '_GN_essential_gene_length.txt', 'w')
            f_gn_non_essential_gene_length = open(str(self.ExpName) +  '_GN_non_essential_gene_length.txt', 'w')

            self.trainingData, self.validationData, self.testingData, _ = combineAndSplitData(self.EssentialGeneFeatTable,
                                                                                           self.NonEssentialGeneFeatTable,
                                                                                           trainingProp,
                                                                                           f_gn_essential_gene_length,
                                                                                           f_gn_non_essential_gene_length)
            f_gn_essential_gene_length.close()
            f_gn_non_essential_gene_length.close()

        elif option == '-cl':
            self.TrainMCLEssentialGeneFeatTable = getGeneFeatTable(feat, self.TrainMCLEssentialGeneNameDict,
                                                                   classLabel=1)
            self.TrainMCLNonEssentialGeneFeatTable = getGeneFeatTable(feat, self.TrainMCLNonEssentialGeneNameDict,
                                                                      classLabel=0)

            self.ValidMCLEssentialGeneFeatTable = getGeneFeatTable(feat, self.ValidMCLEssentialGeneNameDict,
                                                                   classLabel=1)
            self.ValidMCLNonEssentialGeneFeatTable = getGeneFeatTable(feat, self.ValidMCLNonEssentialGeneNameDict,
                                                                      classLabel=0)

            self.TestMCLEssentialGeneFeatTable = getGeneFeatTable(feat, self.TestMCLEssentialGeneNameDict, classLabel=1)
            self.TestMCLNonEssentialGeneFeatTable = getGeneFeatTable(feat, self.TestMCLNonEssentialGeneNameDict,
                                                                     classLabel=0)

            self.clusterTrainingData = resizeData(self.TrainMCLEssentialGeneFeatTable,
                                                  self.TrainMCLNonEssentialGeneFeatTable)
            self.clusterValidData = resizeData(self.ValidMCLEssentialGeneFeatTable,
                                               self.ValidMCLNonEssentialGeneFeatTable)
            self.clusterTestData = resizeData(self.TestMCLEssentialGeneFeatTable, self.TestMCLNonEssentialGeneFeatTable)

            self.trainingData = self.clusterTrainingData
            self.validationData = self.clusterValidData
            self.testingData = self.clusterTestData

    # returns the essential gene feature in a numpy matrix format
    def getEssentialGeneFeatMatrix(self):
        return np.array(self.EssentialGeneFeatTable)

    # returns the non essential gene feature in a numpy matrix format
    def getNonEssentialGeneFeatMatrix(self):
        return np.array(self.NonEssentialGeneFeatTable)

    # returns the complete set of data (imbalanced essential/non essential dataset)
    def getCompleteDataset(self):
        return self.completeData

    # returns the reSized dataset (balanced essential/non essential dataset)
    def getCompleteReSizedDataset(self):
        return self.completeResizedData

    # returns training dataset
    def getTrainingData(self):
        return self.trainingData

    # return validation dataset
    def getValidationData(self):
        return self.validationData

    # return testing dataset
    def getTestingData(self):
        return self.testingData

    # return correlated features list
    def getCorrFeats(self):
        return self.corr_feats

    # returns scaled training dataset
    @staticmethod
    def getScaledData(dataMatrix):
        scaler = StandardScaler().fit(dataMatrix)
        return scaler.transform(dataMatrix)


# returns the feature table containing attributed of each gene. The decision of
# returning essential/non essential gene feature table is made with class label parameter
def getGeneFeatTable(feat, geneNameDict, classLabel):
    featList = list()

    if classLabel == 1:
        EssentialGeneLengthDict = feat.getEssentialGeneLengthFeatDict()
        EssentialKmerFeatDict = feat.getEssentialKmerFeatDict()
        EssentialGCFeatDict = feat.getEssentialGCContentFeatDict()
        EssentialCIARCSUFeatDict = feat.getEssentialCAIRCSUFeatDict()
        EssentialProteinFeatDict = feat.getEssentialProteinFeatDict()

        for geneName in EssentialKmerFeatDict:
            if geneName in geneNameDict:
                attributeList = list()
                attributeList.append(EssentialGeneLengthDict[geneName])
                attributeList.extend(EssentialKmerFeatDict[geneName])
                attributeList.extend(EssentialGCFeatDict[geneName])
                attributeList.extend(EssentialCIARCSUFeatDict[geneName])
                attributeList.extend(EssentialProteinFeatDict[geneName])
                attributeList.append(classLabel)

                featList.append(attributeList)

    if classLabel != 1:
        NonEssentialGeneLengthDict = feat.getNonEssentialGeneLengthFeatDict()
        NonEssentialKmerFeatDict = feat.getNonEssentialKmerFeatDict()
        NonEssentialGCFeatDict = feat.getNonEssentialGCContentFeatDict()
        NonEssentialCIARCSUFeatDict = feat.getNonEssentialCAIRCSUFeatDict()
        NonEssentialProteinFeatDict = feat.getNonEssentialProteinFeatDict()

        for geneName in NonEssentialKmerFeatDict:
            if geneName in geneNameDict:
                attributeList = list()
                attributeList.append(NonEssentialGeneLengthDict[geneName])
                attributeList.extend(NonEssentialKmerFeatDict[geneName])
                attributeList.extend(NonEssentialGCFeatDict[geneName])
                attributeList.extend(NonEssentialCIARCSUFeatDict[geneName])
                attributeList.extend(NonEssentialProteinFeatDict[geneName])
                attributeList.append(classLabel)

                featList.append(attributeList)

    return featList


def resizeData(table1, table2):
    matrix1 = np.array(table1)
    matrix2 = np.array(table2)

    matrix1Row, matrix1Col = matrix1.shape
    matrix2Row, matrix2Col = matrix2.shape

    sampleSize = matrix1Row if matrix1Row <= matrix2Row else matrix2Row
    numSampleToSelect = (sampleSize * 95) / 100

    reSizedMatrix1 = matrix1[np.random.choice(matrix1Row, numSampleToSelect, replace=False), :]
    reSizedMatrix2 = matrix2[np.random.choice(matrix2Row, numSampleToSelect, replace=False), :]

    return np.vstack((reSizedMatrix1, reSizedMatrix2))
