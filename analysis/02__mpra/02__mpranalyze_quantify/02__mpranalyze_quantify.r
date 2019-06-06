
# install MPRAnalyze
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("MPRAnalyze", version = "3.8")

# install RCurl -- not installing separately was causing errors
install.packages("RCurl")

# load the mpranalyze package
library(MPRAnalyze)

dna_counts_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_counts.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)

# since we only have 1 dna replicate -- add another so code doesn't crash (expects matrix)
dna_counts_depth["dna_2"] <- dna_counts_depth["dna_1"]

row.names(dna_counts_depth) <- dna_counts_depth$element
dna_counts_depth <- dna_counts_depth[ , !(names(dna_counts_depth) %in% c("element")), drop=FALSE]
dna_counts_depth <- as.matrix(dna_counts_depth)

rna_counts_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/rna_counts.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)
row.names(rna_counts_depth) <- rna_counts_depth$element
rna_counts_depth <- rna_counts_depth[ , !(names(rna_counts_depth) %in% c("element")), drop=FALSE]
rna_counts_depth <- as.matrix(rna_counts_depth)

dna_cols_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_col_ann.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)
names(dna_cols_depth) <- c("id", "condition", "sample")

# add second row to dna_cols_depth
row2 <- data.frame(id="dna_2", condition="dna", sample="2")
dna_cols_depth <- rbind(dna_cols_depth, row2)
row.names(dna_cols_depth) <- dna_cols_depth$id

rna_cols_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/rna_col_ann.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)
names(rna_cols_depth) <- c("id", "condition", "sample")
row.names(rna_cols_depth) <- rna_cols_depth$id
dna_cols_depth

# make sure everything is a factor
dna_cols_depth$sample <- as.factor(dna_cols_depth$sample)
rna_cols_depth$sample <- as.factor(rna_cols_depth$sample)
dna_cols_depth$condition <- as.factor(dna_cols_depth$condition)
rna_cols_depth$condition <- as.factor(rna_cols_depth$condition)

dna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_counts.mpranalyze.for_quantification.txt", sep="\t", header=TRUE)
row.names(dna_counts) <- dna_counts$element
dna_counts <- dna_counts[ , !(names(dna_counts) %in% c("element"))]
dna_counts <- as.matrix(dna_counts)

rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/rna_counts.mpranalyze.for_quantification.txt", sep="\t", header=TRUE)
row.names(rna_counts) <- rna_counts$element
rna_counts <- rna_counts[ , !(names(rna_counts) %in% c("element"))]
rna_counts <- as.matrix(rna_counts)

dna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_col_ann.mpranalyze.for_quantification.txt", sep="\t", header=TRUE)
row.names(dna_cols) <- dna_cols$X
rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/rna_col_ann.mpranalyze.for_quantification.txt", sep="\t", header=TRUE)
row.names(rna_cols) <- rna_cols$X

# make sure everything is a factor
dna_cols$barcode <- as.factor(dna_cols$barcode)
rna_cols$barcode <- as.factor(rna_cols$barcode)
dna_cols$sample <- as.factor(dna_cols$sample)
rna_cols$sample <- as.factor(rna_cols$sample)
dna_cols$condition <- as.factor(dna_cols$condition)
rna_cols$condition <- as.factor(rna_cols$condition)

ctrls <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/ctrl_status.mpranalyze.for_quantification.txt", sep="\t", header=TRUE)
ctrls <- as.logical(ctrls$ctrl_status)

head(rna_cols)

# create MPRA object
depth_obj <- MpraObject(dnaCounts = dna_counts_depth, rnaCounts = rna_counts_depth, 
                        dnaAnnot = dna_cols_depth, rnaAnnot = rna_cols_depth)

# estimate depth factors using uq -- here, a sample/condition pair == 1 library
depth_obj <- estimateDepthFactors(depth_obj, lib.factor = c("sample", "condition"),  depth.estimator='uq',
                                  which.lib = "both")

rna_depths <- rnaDepth(depth_obj)
rna_depths

rna_cols_depth

# first need to set the dnadepths and rnadepths manually
dna_cols$depth <- rep(1, nrow(dna_cols))

# note 13 will change depending how many barcodes there are per element
rna_cols$depth <- c(rep(rna_depths[1], 13), rep(rna_depths[2], 13), rep(rna_depths[3], 13))

# create MPRA object
obj <- MpraObject(dnaCounts = dna_counts, rnaCounts = rna_counts, 
                  dnaAnnot = dna_cols, rnaAnnot = rna_cols, 
                  controls = ctrls)

# set depth factors manually
obj <- setDepthFactors(obj, dnaDepth = dna_cols$depth, rnaDepth = rna_cols$depth)

# analyze quantification in unpaired DNA library
obj <- analyzeQuantification(obj = obj, 
                             dnaDesign = ~ barcode,
                             rnaDesign = ~ condition)

# extract alphas
alpha <- getAlpha(obj, by.factor = "condition")
head(alpha)

# test against negative controls
res.human <- testEmpirical(obj = obj, statistic = alpha$HUES64)
summary(res.human)

# test against negative controls
res.mouse <- testEmpirical(obj = obj, statistic = alpha$mESC)
summary(res.mouse)

alpha$HUES64_pval <- res.human$pval.mad
alpha$mESC_pval <- res.mouse$pval.mad
head(alpha)

# histogram for negative controls
hist(alpha[ctrls,]$HUES64_pval)

# histogram for negative controls
hist(alpha[ctrls,]$mESC_pval)

# histogram for TSSs
hist(alpha[!ctrls,]$HUES64_pval)

# histogram for TSSs
hist(alpha[!ctrls,]$mESC_pval)

# correct for multiple testing
alpha$HUES64_padj <- p.adjust(alpha$HUES64_pval, method = "fdr")
alpha$mESC_padj <- p.adjust(alpha$mESC_pval, method = "fdr")
head(alpha)

write.table(alpha, file = "../../../data/02__mpra/02__activs/alpha_per_elem.quantification.txt", sep = "\t",
            quote = FALSE)
