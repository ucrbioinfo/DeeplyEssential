import pandas as pd
from Bio import SeqIO


class ReadFiles(object):
    """docstring for ReadFiles"""

    def __init__(self, param):
        super(ReadFiles, self).__init__()

        self.dataset_file = param.getDatasetFilePath()

        self.essential_gene_seq_file = param.getEssentialGeneSeqFilePath()
        self.essential_gene_anno_file = param.getEssentialGeneAnnotFilePath()
        self.essential_aa_file = param.getEssentialAAFilePath()
        self.essential_hit_file = param.getEsentialHitFilePath()

        self.non_essential_gene_seq_file = param.getNonEssentialGeneSeqFilePath()
        self.non_essential_gene_anno_file = param.getNonEssentialGeneAnnotFilePath()
        self.non_essential_aa_file = param.getNonEssentialAAFilePath()
        self.non_essential_hit_file = param.getNonEssentialHitFilePath()

        self.combined_mcl_file = param.getCombinedMCLFilePath()

        # dataset related attributes
        self.datasetInfoDict = dict()
        self.completeOrganismNameList = list()
        self.completeOrganismAccessionList = list()
        self.gramPositiveOrganismNameList = list()
        self.gramPositiveOrganismAccessionList = list()
        self.gramNegativeOrganismNameList = list()
        self.gramNegativeOrganismAccessionList = list()

        # gene annotation related attributes
        self.essentialAnnotationInfoDict = dict()
        self.nonEssentialAnnotationInfoDict = dict()

        self.accessionWiseEssentialGeneCount = dict()
        self.accessionWiseNonEssentialGeneCount = dict()

        # gene seq related attributes
        self.essentialGeneSeqDict = dict()
        self.nonEssentialGeneSeqDict = dict()

        # protein seq related attributes
        self.essentialProteinSeqDict = dict()
        self.nonEssentialProteinSeqDict = dict()

        # mcl related attributes
        self.essentialMCLDict = dict()
        self.nonEssentialMCLDict = dict()

        # combined MCL data info
        self.combinedMCLDict = processCombinedMCLInfo(self.combined_mcl_file)

        # hit related attributes
        self.essentialHitDict = dict()
        self.nonEssentialHitDict = dict()

        # combine essential non essential cluster info
        self.clusterStatInfoDict = getClusterStatistics(self.combinedMCLDict)

    # returns the complete dataset information in a dictionary
    def getDatasetFileInfo(self):

        DataInfo = pd.read_table(self.dataset_file, sep='\t', header='infer')
        for index, row in DataInfo.iterrows():
            self.datasetInfoDict[row['NCBI_Accession_ID']] = [row['Organism'], row['Abbr.'],
                                                              row['GP/GN']]  # need to change the dataset file

        return self.datasetInfoDict

    # returns the complete list of names of the organisms
    def getCompleteListOrganismsName(self):
        for acc_id in self.datasetInfoDict:
            self.completeOrganismNameList.append(self.datasetInfoDict[acc_id][0])

        return self.completeOrganismNameList

    # returns the complete list of accession IDs of the organisms
    def getCompleteListOrganismAccession(self):
        for acc_id in self.datasetInfoDict:
            self.completeOrganismAccessionList.append(acc_id)

        return self.completeOrganismAccessionList

    # returns the list of gram positive organism name
    def getGramPositiveOrganismName(self):
        for acc_id in self.datasetInfoDict:
            if self.datasetInfoDict[acc_id][2] == '+':
                self.gramPositiveOrganismNameList.append(self.datasetInfoDict[acc_id][0])

        return self.gramPositiveOrganismNameList

    # returns the list of gram positive organism accession IDs
    def getGramPositiveOrganismAccession(self):
        for acc_id in self.datasetInfoDict:
            if self.datasetInfoDict[acc_id][2] == '+':
                self.gramPositiveOrganismAccessionList.append(acc_id)

        return self.gramPositiveOrganismAccessionList

    # returns the list of gram negative organism names
    def getGramNegativeOrganismName(self):
        for acc_id in self.datasetInfoDict:
            if self.datasetInfoDict[acc_id][2] == '-':
                self.gramNegativeOrganismNameList.append(self.datasetInfoDict[acc_id][0])

        return self.gramNegativeOrganismNameList

    # returns the list of gram negative organisms accession IDs
    def getGramNegativeOrganismAccession(self):
        for acc_id in self.datasetInfoDict:
            if self.datasetInfoDict[acc_id][2] == '-':
                self.gramNegativeOrganismAccessionList.append(acc_id)

        return self.gramNegativeOrganismAccessionList

    # returns the complete information of essential gene name and the gene sequence
    def getEssentialGeneSeqInfo(self):
        fastaSequences = SeqIO.parse(open(self.essential_gene_seq_file), 'fasta')
        for fasta in fastaSequences:
            name, sequence = fasta.id, str(fasta.seq)
            if "Not" in sequence:
                continue
            self.essentialGeneSeqDict[name] = sequence

        return self.essentialGeneSeqDict

    # returns the essential gene annotation information
    # need to rewrite the annotation file, change header
    def getEssentialGeneAnnoInfo(self):
        AnnotationInfo = pd.read_table(self.essential_gene_anno_file, sep='\t', header='infer')
        for index, row in AnnotationInfo.iterrows():
            self.essentialAnnotationInfoDict[row['#Gene_Name']] = [row['#COG'], row['#Organism'], row['#Conditions']]

            if row['#Conditions'] not in self.accessionWiseEssentialGeneCount:
                self.accessionWiseEssentialGeneCount[row['#Conditions']] = 1
            else:
                self.accessionWiseEssentialGeneCount[row['#Conditions']] += 1

        # for plot
        fopen_ess_gene_count = open('Accession_wise_ess_gene_count.txt', 'w')
        for accession in self.accessionWiseEssentialGeneCount:
            fopen_ess_gene_count.write(str(accession) + '\t' + str(self.accessionWiseEssentialGeneCount[accession]) + '\n')

        fopen_ess_gene_count.close()

        return self.essentialAnnotationInfoDict

    # returns the essential gene name and protein sequence
    def getEssentialProteinInfo(self):
        fastaSequences = SeqIO.parse(open(self.essential_aa_file), 'fasta')
        for fasta in fastaSequences:
            name, sequence = fasta.id, str(fasta.seq)
            self.essentialProteinSeqDict[name] = sequence

        return self.essentialProteinSeqDict

    # returns the MCL information of the essential genes
    def getEssentialHitInfo(self):
        HitInfo = pd.read_table(self.essential_hit_file, sep='\t', header='infer')
        for index, row in HitInfo.iterrows():
            gene = row['Query'].split('>')[1]
            self.essentialHitDict[gene] = [row['E-Value'], row['Bitscore'], row['Short_name']]

        return self.essentialHitDict

    # returns the complete information of non essential gene name and the gene sequence
    def getNonEssentialGeneSeqInfo(self):
        fastaSequences = SeqIO.parse(open(self.non_essential_gene_seq_file), 'fasta')
        for fasta in fastaSequences:
            name, sequence = fasta.id, str(fasta.seq)
            if "Not" in sequence:
                continue
            self.nonEssentialGeneSeqDict[name] = sequence

        return self.nonEssentialGeneSeqDict

    # returns the non essential gene annotation information
    # need to rewrite the annotation file, change header
    def getNonEssentialGeneAnnoInfo(self):
        AnnotationInfo = pd.read_table(self.non_essential_gene_anno_file, sep='\t', header='infer')
        for index, row in AnnotationInfo.iterrows():
            self.nonEssentialAnnotationInfoDict[row['#Gene_Name']] = [row['#COG'], row['#Organism'], row['#Conditions']]

            if row['#Conditions'] not in self.accessionWiseNonEssentialGeneCount:
                self.accessionWiseNonEssentialGeneCount[row['#Conditions']] = 1
            else:
                self.accessionWiseNonEssentialGeneCount[row['#Conditions']] += 1

        # for plot
        # print len(self.accessionWiseNonEssentialGeneCount)
        fopen_ness_gene_count = open('Accession_wise_ness_gene_count.txt', 'w')
        for accession in self.accessionWiseNonEssentialGeneCount:
            fopen_ness_gene_count.write(str(accession) + '\t' + str(self.accessionWiseNonEssentialGeneCount[accession]) + '\n')

        fopen_ness_gene_count.close()

        return self.nonEssentialAnnotationInfoDict

    # returns the essential gene name and protein sequence
    def getNonEssentialProteinInfo(self):
        fastaSequences = SeqIO.parse(open(self.non_essential_aa_file), 'fasta')
        for fasta in fastaSequences:
            name, sequence = fasta.id, str(fasta.seq)
            self.nonEssentialProteinSeqDict[name] = sequence

        return self.nonEssentialProteinSeqDict

    # returns the MCL information of the non essential genes
    def getNonEssentialHitInfo(self):
        HitInfo = pd.read_table(self.non_essential_mcl_file, sep='\t', header='infer')
        for index, row in HitInfo.iterrows():
            gene = row['Query'].split('>')[1]
            self.nonEssentialHitDict[gene] = [row['E-Value'], row['Bitscore'], row['Short_name']]

        return self.nonEssentialHitDict

    # return combined MCL dict
    def getCombinedMCLInfo(self):
        return self.combinedMCLDict

    # returns cluster (MCL) statistics
    def getMCLStatistics(self):
        return self.clusterStatInfoDict

# return non_essential gene MCL information
def processCombinedMCLInfo(mcl_file):
    combinedMCLDict = dict()
    clusterID = 1
    with open(mcl_file) as f:
        for line in f:
            line = line.strip()
            cols = line.split()

            combinedMCLDict[clusterID] = cols[1:]
            clusterID += 1

    return combinedMCLDict


# returns essential non essential MCL combined statistics
def getClusterStatistics(combinedMCLDict):
    clusterStatInfoDict = dict()

    for cluster in combinedMCLDict:
        essentialGeneCount, nonEssentialGeneCount = 0, 0
        geneList = combinedMCLDict[cluster]
        for i in range(0, len(geneList)):
            if 'DEG' in geneList[i]:
                essentialGeneCount += 1
            elif 'DNEG' in geneList[i]:
                nonEssentialGeneCount += 1

        essentialGenePercentage = float(essentialGeneCount) / (essentialGeneCount + nonEssentialGeneCount)
        nonEssentialGenePercentage = float(nonEssentialGeneCount) / (essentialGeneCount + nonEssentialGeneCount)

        clusterStatInfoDict[cluster] = [essentialGeneCount,
                                        nonEssentialGeneCount,
                                        essentialGenePercentage,
                                        nonEssentialGenePercentage,
                                        abs(essentialGenePercentage - nonEssentialGenePercentage)
                                        ]
    return clusterStatInfoDict
