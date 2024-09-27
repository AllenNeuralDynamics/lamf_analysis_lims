exp_fire_plot <- function (data, anno, genes, grouping, group_order=NULL, log_scale=TRUE, 
          normalize_rows=FALSE, colorset=NULL, top_values="lowest", 
          font_size=7, label_height=25, max_width=10, return_type="plot", reformat=FALSE) 
{
  # reversing order of genes list
  genes <- rev(genes)  
  # making cluster_*** labels
  group_cols <- group_columns(grouping)
  # if data is in dplyr format, convert to separate anno and 
  if (reformat){
    # data_tidy <- data # in case I want to figure out how to do everything using dplyr formatting
    anno <- as.data.frame(data %>%
                            select(-gene,-exp) %>%
                            distinct())
    data <- data %>%
      select(sample_name,gene,exp) %>%
      pivot_wider(names_from=gene,values_from=exp)
  }
  # making sure data and anno samples are all in the same order, unless there's a specified group order 
  gene_data <- filter_gene_data(data, genes, anno, group_cols, group_order, "sample_name")
  # putting anno in the group order, if specified?
  if (!is.null(group_order)) {
    anno <- anno[anno[[group_cols$id]] %in% group_order,   
    ]
  }
  # get the maximum value for each gene
  max_vals_unscaled <- max_vals <- purrr::map_dbl(genes, function(x) {max(data[[x]], na.rm=TRUE)})  
  names(max_vals_unscaled) <- genes
  # putting anno and data in the same data frame
  plot_data <- left_join(anno, gene_data, by="sample_name") 
  # scale data
  if (log_scale) {
    plot_data <- scale_gene_data(plot_data, genes, scale_type="log10")
  }
  # adding the order for each group to the data frame
  plot_data <- add_group_xpos(plot_data,group_cols=group_cols, 
                              group_order=group_order)
  # counting how many genes, groups, and samples
  n_stats <- get_n_stats(plot_data, group_cols, genes)
  
  # assigns visual aspects for each cluster label
  header_labels <- build_header_labels(data=plot_data, 
                                       grouping=grouping, 
                                       group_order=group_order, 
                                       ymin=n_stats$genes+1, 
                                       label_height=label_height, 
                                       label_type="simple")
  # makes the size and basics of heat map plot - blank for now
  p <- ggplot(data) + 
    scale_fill_identity() +
    theme_classic(base_size=font_size) +
    theme(axis.text=element_text(size=rel(1), face="italic"), 
          axis.ticks=element_blank(), 
          axis.line=element_blank(), 
          axis.title=element_blank(), 
          axis.text.x=element_blank()) + 
    scale_x_continuous(expand=c(0, 0)) + 
    scale_y_continuous(expand=c(0,0),
                       breaks=1:n_stats$genes + 0.45,
                       labels=genes)
  
  for (i in seq_along(genes)) {
    # what gene plotting
    gene <- genes[i]
    # getting gene's vlues
    gene_values <- plot_data %>% 
      select(one_of("sample_name",gene))
    # getting heat map colors
    gene_colors <- data_df_to_colors(gene_values, 
                                     value_cols=gene,
                                     per_col=normalize_rows, 
                                     colorset=colorset)
    names(gene_colors)[names(gene_colors) == gene] <- "plot_fill"
    
    if (top_values == "highest"){
      rect_data <- plot_data %>% 
        left_join(gene_colors,by="sample_name") %>% 
        arrange_(gene) %>% 
        group_by(xpos) %>% 
        mutate(group_n=n()) %>%
        mutate(xmin=xpos-0.5, 
               xmax=xpos+0.5, 
               ymin=seq(i,i+1,length.out=group_n[1]+1)[-(group_n[1]+1)],
               ymax=seq(i,i+1,length.out=group_n[1]+1)[-1])
    }else{
      arr_gene <- paste0("-",gene)
      rect_data <- plot_data %>%
        left_join(gene_colors,by="sample_name") %>%
        arrange_(arr_gene) %>% 
        group_by(xpos) %>% 
        mutate(group_n=n()) %>% 
        mutate(xmin=xpos-0.5,
               xmax=xpos+0.5,
               ymin=seq(i,i+1,length.out=group_n[1]+1)[-(group_n[1]+1)],
               ymax=seq(i,i+1,length.out=group_n[1]+1)[-1])
    }
    p <- p + geom_rect(data=rect_data,aes(xmin=xmin,
                                            xmax=xmax, 
                                            ymin=ymin, 
                                            ymax=ymax, 
                                            fill=plot_fill))
  }
  header_labels <- build_header_labels(data=plot_data, 
                                       grouping=grouping, 
                                       ymin=n_stats$genes + 1, 
                                       label_height=label_height, 
                                       label_type="simple")

  p <- p + geom_rect(data = header_labels, 
                     aes(xmin = xmin,xmax = xmax, ymin = ymin, ymax = ymax, fill = color)) + 
    geom_text(data = header_labels, 
              aes(x = (xmin + xmax)/2,y = ymin + 0.05, label = label), 
              angle = 90, 
              vjust = 0.35, 
              hjust = 0, 
              size = pt2mm(font_size),
              color="white")

  
  max_val_dfs <- build_max_dfs(n_stats, 
                               width_stat="groups", 
                               max_vals_unscaled, 
                               max_width)
  
  p <- ggplot_max_vals(p, 
                       n_stats=n_stats, 
                       width_stat="groups", 
                       max_val_dfs=max_val_dfs, 
                       font_size=font_size)
  
  if (return_type == "plot"){
    return(p)
  }
  else if(return_type == "data"){
    return(list(plot_data=plot_data,
                header_labels=header_labels, 
                header_polygons=header_polygons, 
                max_val_dfs=max_val_dfs, 
                n_stats=n_stats))
  }else if(return_type == "both"){
    return(list(plot=p, 
                plot_data=plot_data, 
                header_labels=header_labels, 
                max_val_dfs=max_val_dfs, 
                n_stats=n_stats))
  }
}
