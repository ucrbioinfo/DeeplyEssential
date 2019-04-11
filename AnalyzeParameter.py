class ParameterProcessing(object):

	def __init__(self, essential_path, non_essential_path, dataset, mclFile):
		super(ParameterProcessing, self).__init__()
		self.essential_path = essential_path
		self.non_essential_path = non_essential_path
		self.dataset = dataset
		self.combined_mcl_file_path = mclFile

	# returns the file path for the essential gene sequence file
	def getEssentialGeneSeqFilePath(self):
		return self.essential_path + str('degseq-p.dat')

	# returns the file path for the essential gene protein seq file
	def getEssentialAAFilePath(self):
		return self.essential_path + str('degaa-p.dat')

	# returns the file path for the essential gene annotation file
	def getEssentialGeneAnnotFilePath(self):
		return self.essential_path + str('degannotation-p.dat')

	# returns the file path for the essential gene MCL file
	# def getEssentialMCLFilePath(self):
	# 	return self.essential_path + str('essential_groups.txt')

	# returns the file path for essential gene hit file
	def getEsentialHitFilePath(self):
		return self.essential_path + str('Essential_gene_hit_data.txt')

	# returns the file path for the non_essential gene sequence file
	def getNonEssentialGeneSeqFilePath(self):
		return self.non_essential_path + str('degseq-np.dat')

	# returns the file path for the non_essential gene protein seq file
	def getNonEssentialAAFilePath(self):
		return self.non_essential_path + str('degaa-np.dat')

	# returns the file path for the non_essential gene annotation file
	def getNonEssentialGeneAnnotFilePath(self):
		return self.non_essential_path + str('degannotation-np.dat')

	# returns the file path for non_essential gene hit file
	def getNonEssentialHitFilePath(self):
		return self.non_essential_path + str('NonEssential_gene_hit_data.txt')

	# returns the file path for combined MCL file
	def getCombinedMCLFilePath(self):
		return self.combined_mcl_file_path

	# returns the file path for the dataset to be used in the experiment
	def getDatasetFilePath(self):
		return self.dataset

