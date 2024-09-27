fractionCorrectWithGenes_v2 <- function (orderedGenes, mapDat, medianDat, clustersF, verbose = FALSE, 
          plot = FALSE, return = TRUE, ...) 
{
  numGn <- 2:length(orderedGenes)
  frac <- rep(0, length(orderedGenes))
  for (i in numGn) {
    gns <- orderedGenes[1:i]
    corMapTmp <- suppressWarnings(corTreeMapping(mapDat = mapDat, 
                                                 medianDat = medianDat, genesToMap = gns))
    corMapTmp[is.na(corMapTmp)] <- 0
    topLeafTmp <- getTopMatch(corMapTmp)
    frac[i] <- 100 * mean(topLeafTmp[, 1] == clustersF)
  }
  frac[is.na(frac)] <- 0
  # if (plot) {
  #   genes <- names(frac)
  #   numGn <- 1:length(frac)
  #   pl <- plot(numGn, frac, type = "l", col = "grey", xlab = "Number of genes in panel",  ylab = "Percent of nuclei correctly mapping",
  #              main = "All clusters gene panel",  ylim = c(-10, 100),  lwd = 5)
  #   abline(h = (-2:20) * 5, lty = "dotted", col = "grey")
  #   abline(h = 0, col = "black", lwd = 2)
  #   text(numGn, frac, genes, srt = 90)
  # 
  # }
  # plt_info <- list(pl,frac)
  if (return) {
    return(frac)
  }
}
