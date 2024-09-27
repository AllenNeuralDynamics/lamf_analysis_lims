## ----install packages, eval=FALSE---------------------------------------------
#  install.packages("devtools")
#  devtools::install_github("AllenInstitute/scrattch.vis")  # For plotting
#  devtools::install_github("AllenInstitute/tasic2016data") # For our data example

## ----load libraries-----------------------------------------------------------
suppressPackageStartupMessages({
  library(mfishtools)    # This library!
  library(gplots)        # This is for plotting gene panels only.
  library(scrattch.vis)  # This is for plotting gene panels only.
  library(matrixStats)   # For rowMedians function, which is fast
  library(tasic2016data) # For the data
})
options(stringsAsFactors = FALSE)  # IMPORTANT
print("Libraries loaded.")

## ----load tasic data----------------------------------------------------------
annotations <- tasic_2016_anno
counts      <- tasic_2016_counts
rpkm        <- tasic_2016_rpkm
annotations <- annotations[match(colnames(counts),annotations$sample_name),]  # Put them in the correct order

## ----define variables---------------------------------------------------------
clusterType = annotations$broad_type 
includeClas = "GABA-ergic Neuron"  # In this analysis, we are only considering interneurons
excludeClas = sort(setdiff(clusterType,includeClas))
gliaClas    = setdiff(excludeClas,"Glutamatergic Neuron") 
kpSamp      = !is.element(clusterType,excludeClas)
anno        = annotations[kpSamp,]
cl          = annotations$primary_type_label
names(cl)   = annotations$sample_name
kpClust     = sort(unique(cl[kpSamp]))
gliaClust   = sort(unique(cl[is.element(clusterType,gliaClas)]))

## ----convert to log2----------------------------------------------------------
normDat = log2(rpkm+1)
#sf      = colSums(counts)/10^6
#cpms    = t(t(counts)/sf)
#normDat = log2(cpms+1)

## ----calculate proportions and medians----------------------------------------
exprThresh = 1
medianExpr = do.call("cbind", tapply(names(cl), cl, function(x) rowMedians(normDat[,x]))) 
propExpr   = do.call("cbind", tapply(names(cl), cl, function(x) rowMeans(normDat[,x]>exprThresh))) 
rownames(medianExpr) <- rownames(propExpr) <- genes <- rownames(normDat)  

## ----filter genes for panel selection-----------------------------------------
startingGenePanel <-  c("Gad1","Slc32a1","Pvalb","Sst","Vip")
runGenes <- NULL
runGenes <- filterPanelGenes(
  summaryExpr = 2^medianExpr-1,  # medians (could also try means); We enter linear values to match the linear limits below
  propExpr    = propExpr,    # proportions
  onClusters  = kpClust,     # clusters of interest for gene panel
  offClusters = gliaClust,   # clusters to exclude expression
  geneLengths = NULL,        # vector of gene lengths (not included here)
  startingGenes  = startingGenePanel,  # Starting genes (from above)
  numBinaryGenes = 250,      # Number of binary genes (explained below)
  minOn     = 10,   # Minimum required expression in highest expressing cell type
  maxOn     = 500,  # Maximum allowed expression
  maxOff    = 50,   # Maximum allowed expression in off types (e.g., aviod glial expression)
  minLength = 960,  # Minimum gene length (to allow probe design; ignored in this case)
  fractionOnClusters = 0.5,  # Max fraction of on clusters (described above)
  excludeGenes    = NULL,    # Genes to exclude.  Often sex chromosome or mitochondrial genes would be input here.
  excludeFamilies = c("LOC","Fam","RIK","RPS","RPL","\\-","Gm","Rnf","BC0")) # Avoid LOC markers, in this case

## ----identify binary markers--------------------------------------------------
corDist         <- function(x) return(as.dist(1-cor(x)))
clusterDistance <- as.matrix(corDist(medianExpr[runGenes,kpClust]))
print(dim(clusterDistance))

## ----build gene panels--------------------------------------------------------
fishPanel <- buildMappingBasedMarkerPanel(
  mapDat        = normDat[runGenes,kpSamp],     # Data for optimization
  medianDat     = medianExpr[runGenes,kpClust], # Median expression levels of relevant genes in relevant clusters
  clustersF     = cl[kpSamp],                   # Vector of cluster assignments
  panelSize     = 30,                           # Final panel size
  currentPanel  = startingGenePanel,            # Starting gene panel
  subSamp       = 15,                           # Maximum number of cells per cluster to include in analysis (20-50 is usually best)
  optimize      = "CorrelationDistance",        # CorrelationDistance maximizes the cluster distance as described
  clusterDistance = clusterDistance,            # Cluster distance matrix
  percentSubset = 50                            # Only consider a certain percent of genes each iteration to speed up calculations (in most cases this is not recommeded)
)       


## ----panel gene plot in subset, fig.width=7,fig.height=10---------------------
plotGenes <- fishPanel
plotData  <- cbind(sample_name = colnames(rpkm), as.data.frame(t(rpkm[plotGenes,])))
clid_inh  <- 1:23  # Cluster IDs for inhibitory clusters

# violin plot example.  Could be swapped with fire, dot, bar, box plot, heatmap, Quasirandom, etc.
sample_fire_plot(data = plotData, anno = annotations, genes = plotGenes, grouping = "primary_type", 
                 log_scale=TRUE, max_width=15, label_height = 8, group_order = clid_inh)

## ----panel gene plot across all types, fig.width=14,fig.height=10-------------
sample_fire_plot(data = plotData, anno = annotations, genes = plotGenes, grouping = "primary_type", 
                 log_scale=TRUE, max_width=15, label_height = 8)

## ----dsiplay fraction correctly mapped----------------------------------------
assignedCluster <- suppressWarnings(getTopMatch(corTreeMapping(mapDat = normDat[runGenes,kpSamp], 
                   medianDat=medianExpr[runGenes,kpClust], genesToMap=fishPanel)))
print(paste0("Percent correctly mapped: ",signif(100*mean(as.character(assignedCluster$TopLeaf)==cl[kpSamp],na.rm=TRUE),3),"%"))

## ----plot fraction correctly mapped,fig.width=9,fig.height=6------------------
fractionCorrectWithGenes(fishPanel,normDat[,kpSamp],medianExpr[runGenes,kpClust],cl[kpSamp],
                         main="Mapping quality for different numbers of included genes",return=FALSE)

## ----cluster confusion, fig.height=8,fig.width=8------------------------------
membConfusionProp  <- getConfusionMatrix(cl[kpSamp],assignedCluster[,1],TRUE)
clOrd <- (annotations$primary_type_label[match(clid_inh,annotations$primary_type_id)])  # Cluster order
heatmap.2(pmin(membConfusionProp,0.25)[clOrd,clOrd],Rowv=FALSE,Colv=FALSE,trace="none",dendrogram="none",
          margins=c(16,16),main="Confusion Matrix")

