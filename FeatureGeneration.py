import math
import itertools

# amino acid dictionary
AminoAcidDictionary = {
    'A': 0, 'R': 0, 'N': 0, 'D': 0,
    'C': 0, 'Q': 0, 'E': 0, 'G': 0,
    'H': 0, 'I': 0, 'L': 0, 'K': 0,
    'M': 0, 'F': 0, 'P': 0, 'S': 0,
    'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# this dictionary contains all codones
CodonsDict = {
    'TTT': 0, 'TTC': 0, 'TTA': 0, 'TTG': 0, 'CTT': 0,
    'CTC': 0, 'CTA': 0, 'CTG': 0, 'ATT': 0, 'ATC': 0,
    'ATA': 0, 'ATG': 0, 'GTT': 0, 'GTC': 0, 'GTA': 0,
    'GTG': 0, 'TAT': 0, 'TAC': 0, 'TAA': 0, 'TAG': 0,
    'CAT': 0, 'CAC': 0, 'CAA': 0, 'CAG': 0, 'AAT': 0,
    'AAC': 0, 'AAA': 0, 'AAG': 0, 'GAT': 0, 'GAC': 0,
    'GAA': 0, 'GAG': 0, 'TCT': 0, 'TCC': 0, 'TCA': 0,
    'TCG': 0, 'CCT': 0, 'CCC': 0, 'CCA': 0, 'CCG': 0,
    'ACT': 0, 'ACC': 0, 'ACA': 0, 'ACG': 0, 'GCT': 0,
    'GCC': 0, 'GCA': 0, 'GCG': 0, 'TGT': 0, 'TGC': 0,
    'TGA': 0, 'TGG': 0, 'CGT': 0, 'CGC': 0, 'CGA': 0,
    'CGG': 0, 'AGT': 0, 'AGC': 0, 'AGA': 0, 'AGG': 0,
    'GGT': 0, 'GGC': 0, 'GGA': 0, 'GGG': 0}

# this dictionary shows which codons encode the same AA
SynonymousCodons = {
    'CYS': ['TGT', 'TGC'],
    'ASP': ['GAT', 'GAC'],
    'SER': ['TCT', 'TCG', 'TCA', 'TCC', 'AGC', 'AGT'],
    'GLN': ['CAA', 'CAG'],
    'MET': ['ATG'],
    'ASN': ['AAC', 'AAT'],
    'PRO': ['CCT', 'CCG', 'CCA', 'CCC'],
    'LYS': ['AAG', 'AAA'],
    'STOP': ['TAG', 'TGA', 'TAA'],
    'THR': ['ACC', 'ACA', 'ACG', 'ACT'],
    'PHE': ['TTT', 'TTC'],
    'ALA': ['GCA', 'GCC', 'GCG', 'GCT'],
    'GLY': ['GGT', 'GGG', 'GGA', 'GGC'],
    'ILE': ['ATC', 'ATA', 'ATT'],
    'LEU': ['TTA', 'TTG', 'CTC', 'CTT', 'CTG', 'CTA'],
    'HIS': ['CAT', 'CAC'],
    'ARG': ['CGA', 'CGC', 'CGG', 'CGT', 'AGG', 'AGA'],
    'TRP': ['TGG'],
    'VAL': ['GTA', 'GTC', 'GTG', 'GTT'],
    'GLU': ['GAG', 'GAA'],
    'TYR': ['TAT', 'TAC']}

EssentialKmerCount = CodonsDict.copy()
NonEssentialKmerCount = CodonsDict.copy()

EssentialGCContent = dict()
NonEssentialGCContent = dict()


class FeatureProcessing(object):

    def __init__(self, read):
        super(FeatureProcessing, self).__init__()

        self.EssentialGeneSeqInfo = read.getEssentialGeneSeqInfo()
        self.EssentialProteinSeqInfo = read.getEssentialProteinInfo()
        self.EssentialAnnotationInfo = read.getEssentialGeneAnnoInfo()

        self.NonEssentialGeneSeqInfo = read.getNonEssentialGeneSeqInfo()
        self.NonEssentialProteinSeqInfo = read.getNonEssentialProteinInfo()
        self.NonEssentialAnnotationInfo = read.getNonEssentialGeneAnnoInfo()

        # MCL info
        self.MCLStatDict = read.getMCLStatistics()
        self.combinedMCLDict = read.getCombinedMCLInfo()

        self.EssentialGCContentFeatDict = dict()
        self.EssentialKmerFeatDict = dict()
        self.EssentialCAIRCSUFeatDict = dict()
        self.EssentialProteinFeatDict = dict()
        self.EssentialGeneSeqLenghtFeatDict = dict()
        self.maxEssentialGeneLength = 0

        self.NonEssentialGCContentFeatDict = dict()
        self.NonEssentialKmerFeatDict = dict()
        self.NonEssentialCAIRCSUFeatDict = dict()
        self.NonEssentialProteinFeatDict = dict()
        self.NonEssentialGeneSeqLengthFeatDict = dict()
        self.maxNonEssentialGeneLength = 0

    # returns a dictionary with essential gene name as key and gene length as values
    def getEssentialGeneLengthFeatDict(self):
        return self.EssentialGeneSeqLenghtFeatDict

    # returns a dictionary with essential gene name as key and binned gc content as values
    def getEssentialGCContentFeatDict(self):
        return self.EssentialGCContentFeatDict

    # returns a dictionary with essential gene name as key and binned kmer freq as values
    def getEssentialKmerFeatDict(self):
        return self.EssentialKmerFeatDict

    # returns a dictionary with essential gene name as key and CAI, MAXRCSU value of the whole sequence
    def getEssentialCAIRCSUFeatDict(self):
        return self.EssentialCAIRCSUFeatDict

    # returns a dictionary with essential gene name as key and a list containing amino acid freq and protein seq length
    def getEssentialProteinFeatDict(self):
        return self.EssentialProteinFeatDict

    # returns the length of the longest essential gene sequence
    def getMaxEssentialGeneLength(self):
        return self.maxEssentialGeneLength

    # returns a dictionary with non essential gene name as key and gene seq length as values
    def getNonEssentialGeneLengthFeatDict(self):
        return self.NonEssentialGeneSeqLengthFeatDict

    # returns a dictionary with non essential gene name as key and binned gc content as values
    def getNonEssentialGCContentFeatDict(self):
        return self.NonEssentialGCContentFeatDict

    # returns a dictionary with non essential gene name as key and binned kmer freq as values
    def getNonEssentialKmerFeatDict(self):
        return self.NonEssentialKmerFeatDict

    # returns a dictionary with non essential gene name as key and CAI, MAXRCSU value of the whole sequence
    def getNonEssentialCAIRCSUFeatDict(self):
        return self.NonEssentialCAIRCSUFeatDict

    # returns a dictionary with non essential gene name as key and a list containing amino acid freq and protein seq length
    def getNonEssentialProteinFeatDict(self):
        return self.NonEssentialProteinFeatDict

    # returns the length of the longest non essential gene sequence
    def getMaxNonEssentialGeneLength(self):
        return self.maxNonEssentialGeneLength

    # return combined MCL dict
    def getCombinedMCLDict(self):
        return self.combinedMCLDict

    # return MCL statistics dict
    def getMCLStatDict(self):
        return self.MCLStatDict

    # this method is the entry point to process all feature
    def getFeatures(self, bins):
        # process and store features for essential genes
        for seqName in self.EssentialGeneSeqInfo:

            # find the gene with maximum length
            if len(self.EssentialGeneSeqInfo[seqName]) > self.maxEssentialGeneLength:
                self.maxEssentialGeneLength = len(self.EssentialGeneSeqInfo[seqName])

            # store gene sequence length
            self.EssentialGeneSeqLenghtFeatDict[seqName] = len(self.EssentialGeneSeqInfo[seqName])

            # get CAI and Max RCSU from the sequence
            CAI, RCSUMax = getCAIRCSU(self.EssentialGeneSeqInfo[seqName])
            self.EssentialCAIRCSUFeatDict[seqName] = [CAI, RCSUMax]

            # get protein seq feature
            proteinFeature = getProteinFeat(self.EssentialProteinSeqInfo[seqName])
            self.EssentialProteinFeatDict[seqName] = proteinFeature

            # split sequence into bin size and get features of each bin
            # kmer frequency of each binned sequence
            # gc content of each binned sequence
            binnedSeqFeat = list()
            binnedGCFeat = list()
            binnedSeqLength = len(self.EssentialGeneSeqInfo[seqName]) / bins
            startPos = 0
            endPos = binnedSeqLength
            for i in range(0, bins):
                if endPos > len(self.EssentialGeneSeqInfo[seqName]):
                    binnedSequence = self.EssentialGeneSeqInfo[seqName][startPos:]
                else:
                    binnedSequence = self.EssentialGeneSeqInfo[seqName][startPos:endPos]
                    startPos = endPos
                    endPos = endPos + binnedSeqLength

                binnedKmerFreqList, binnedKmerDict = getKmerCount(binnedSequence, k = 3)
                binnedGCContent = getGCContent(binnedSequence)

                binnedSeqFeat.extend(binnedKmerFreqList)
                binnedGCFeat.append(binnedGCContent)

                # store all kmer info in essential kmer dictionary
                for kmer in binnedKmerDict:
                    EssentialKmerCount[kmer] = EssentialKmerCount[kmer] + binnedKmerDict[kmer]

                EssentialGCContent[seqName] = binnedGCContent

            self.EssentialKmerFeatDict[seqName] = binnedSeqFeat
            self.EssentialGCContentFeatDict[seqName] = binnedGCFeat

        # process and store features for non essential genes
        for seqName in self.NonEssentialGeneSeqInfo:

            # find the gene with maximum length
            if len(self.NonEssentialGeneSeqInfo[seqName]) > self.maxNonEssentialGeneLength:
                self.maxNonEssentialGeneLength = len(self.NonEssentialGeneSeqInfo[seqName])

            # store gene sequence length
            self.NonEssentialGeneSeqLengthFeatDict[seqName] = len(self.NonEssentialGeneSeqInfo[seqName])

            # get CAI and Max RCSU from the sequence
            CAI, RCSUMax = getCAIRCSU(self.NonEssentialGeneSeqInfo[seqName])
            self.NonEssentialCAIRCSUFeatDict[seqName] = [CAI, RCSUMax]

            # get protein seq feature
            proteinFeature = getProteinFeat(self.NonEssentialProteinSeqInfo[seqName])
            self.NonEssentialProteinFeatDict[seqName] = proteinFeature

            # split sequence into bin size and get features of each bin
            # kmer frequency of each binned sequence
            # gc content of each binned sequence
            binnedSeqFeat = list()
            binnedGCFeat = list()
            binnedSeqLength = len(self.NonEssentialGeneSeqInfo[seqName]) / bins
            startPos = 0
            endPos = binnedSeqLength
            for i in range(0, bins):
                if endPos > len(self.NonEssentialGeneSeqInfo[seqName]):
                    binnedSequence = self.NonEssentialGeneSeqInfo[seqName][startPos:]
                else:
                    binnedSequence = self.NonEssentialGeneSeqInfo[seqName][startPos:endPos]
                    startPos = endPos
                    endPos = endPos + binnedSeqLength

                binnedKmerFreqList, binnedKmerDict = getKmerCount(binnedSequence, k = 3)
                binnedGCContent = getGCContent(binnedSequence)

                binnedSeqFeat.extend(binnedKmerFreqList)
                binnedGCFeat.append(binnedGCContent)

                # store all kmer info in non essential kmer dictionary
                for kmer in binnedKmerDict:
                    NonEssentialKmerCount[kmer] = NonEssentialKmerCount[kmer] + binnedKmerDict[kmer]

                NonEssentialGCContent[seqName] = binnedGCContent

            self.NonEssentialKmerFeatDict[seqName] = binnedSeqFeat
            self.NonEssentialGCContentFeatDict[seqName] = binnedGCFeat

# returns the GC content of a gene sequence
def getGCContent(sequence):
    A, T, C, G = 0, 0, 0, 0

    for i in range(len(sequence)):
        if sequence[i] == 'A':
            A += 1
        elif sequence[i] == 'T':
            T += 1
        elif sequence[i] == 'C':
            C += 1
        elif sequence[i] == 'G':
            G += 1

    return float(G + C) / float(A + T + C + G)


# returns the codon Index dictionary and RCSU max value. This method is a part of getCAIRCSUMax method
def generateCodonIndexandMaxRCSU(sequence):
    codonCountDict = CodonsDict.copy()
    codonIndexDict = dict()

    for i in range(0, len(sequence), 1):
        codon = sequence[i: i + 3]
        if codon in codonCountDict:
            codonCountDict[codon] += 1

    for aminoacid in SynonymousCodons:
        total = 0.0
        rcsu = []
        codons = SynonymousCodons[aminoacid]

        for codon in codons:
            total += codonCountDict[codon]

        for codon in codons:
            denom = float(total) / len(codons)
            if denom > 0:
                rcsu.append(float(codonCountDict[codon]) / denom)
            else:
                rcsu.append(1.)

        rcsuMax = max(rcsu)
        for codonIndex, codon in enumerate(codons):
            codonIndexDict[codon] = float(rcsu[codonIndex]) / rcsuMax

    return codonIndexDict, rcsuMax


# returns the CAI and MAX RCSU value of a sequence
def getCAIRCSU(sequence):
    CAIValue, CAILength = 0, 0
    codonIndexDict, RCSUMax = generateCodonIndexandMaxRCSU(sequence)

    for i in range(0, len(sequence), 3):
        codon = sequence[i: i + 3]
        if codon in codonIndexDict and codon not in ['ATG', 'TGG']:
            CAIValue += math.log(codonIndexDict[codon])
            CAILength += 1

    return float(math.exp(CAIValue)) / (CAILength - 1.0), RCSUMax


# returns protein sequence related feature, amino acid count and protein length
def getProteinFeat(sequence):
    AminoAcidDict = AminoAcidDictionary.copy()

    for i in range(0, len(sequence)):
        if sequence[i] in AminoAcidDict:
            AminoAcidDict[sequence[i]] += 1

    proteinFeat = list()
    for key in AminoAcidDict:
        proteinFeat.append(AminoAcidDict[key])
    proteinFeat.append(len(sequence))

    return proteinFeat


# returns kmer count of the sequence
def getKmerCount(sequence, k):
    string = 'ATCG'
    kMerDict = dict()
    kmerFreqList = list()
    for i in itertools.product(string, repeat=k):
        kmer = ''.join(i)
        kMerDict[kmer] = 0

    # count kmers in the sequence
    for i in range(len(sequence) - k):
        kmer = sequence[i:(i + k)]
        if kmer in kMerDict:
            kMerDict[kmer] += 1

    # storing kmer counts in a list
    for kmer in kMerDict:
        kmerFreqList.append(kMerDict[kmer])

    return kmerFreqList, kMerDict

