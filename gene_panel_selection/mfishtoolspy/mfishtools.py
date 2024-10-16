import numpy as np
import pandas as pd
import re
from scipy.stats import rankdata
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, pdist, squareform


def build_mapping_based_marker_panel(map_data, median_data=None, cluster_call=None, panel_size=50, num_subsample=20, 
                                    max_fc_gene=1000, qmin=0.75, seed=None, current_panel=None, 
                                    panel_min=5, verbose=True, corr_mapping=True, 
                                    optimize="fraction_correct", cluster_distance=None, 
                                    cluster_genes=None, dend=None, percent_gene_subset=100):
    if optimize == "dendrogram_height" and dend is None:
        return "Error: dendrogram not provided"
    
    if median_dat is None:
        median_data = pd.DataFrame()
        cluster_call = pd.Series(cluster_call, index=map_data.columns)
        median_data = map_data.groupby(cluster_call, axis=1).median()
    
    if median_data.index.isnull().any():
        median_data.index = map_data.index

    if optimize == "fraction_correct":
        cluster_distance = None

    if optimize == "correlation_distance":
        if cluster_distance is None:
            cor_dist = lambda x: 1 - np.corrcoef(x)
            if cluster_genes is None:
                cluster_genes = median_data.index
            cluster_genes = list(set(cluster_genes).intersection(set(median_data.index)))
            cluster_distance = pd.DataFrame(cor_dist(median_data.loc[cluster_genes, :].T),
                                            index=median_data.columns, columns=median_data.columns)
        
        if isinstance(cluster_distance, pd.DataFrame):
            cluster_distance = cluster_distance.loc[median_data.columns, median_data.columns].values.flatten()
    
    if optimize == "dendrogram_height":
        # Custom make_LCA_table and get_node_height functions need to be implemented
        lca_table = make_LCA_table(dend)
        cluster_distance = 1 - get_node_height(dend)[lca_table]
        optimize = "correlation_distance"

    # Calculate the gene expression difference between the 100th percentile cluster and the qmin percentile cluster
    # To be used for filtering (if not filtereed before)
    rank_expr_diff = rankdata(median_data.apply(lambda x: np.diff(np.percentile(x, [100 * qmin, 100])), axis=1))
    
    if median_data.shape[0] > max_fc_gene: # filter based on rank_expr_diff
        keep_genes = np.array(median_data.index)[rank_expr_diff <= max_fc_gene] # rankdata rank starts at 1
        map_data = map_data.loc[keep_genes, :]
        median_data = median_data.loc[keep_genes, :]

    panel_min = max(2, panel_min)
    if current_panel is None or len(current_panel) < panel_min:
        panel_min = max(2, panel_min - (len(current_panel) if current_panel else 0))
        current_panel = list(set(current_panel or []).union(set(median_data.index[rank_expr_diff.argsort()[:panel_min]])))
        if verbose:
            print(f"Setting starting panel as: {', '.join(current_panel)}")
    
    if len(current_panel) < panel_size:
        if num_subsample is not None:
            keep_sample = subsample_cells(cluster_call, num_subsample, seed)
            map_data = map_data.loc[:, keep_sample]
            cluster_call = cluster_call[keep_sample]
            num_subsample = None  # Once subsampled, don't subsample again in the next recursion
        
        other_genes = list(set(map_data.index).difference(set(current_panel)))
        if percent_gene_subset < 100:
            if seed is not None:
                np.random.seed(seed + len(current_panel))
            other_genes = np.random.choice(other_genes, size=int(len(other_genes) * percent_gene_subset / 100), replace=False)
        
        match_count = np.zeros(len(other_genes))
        cluster_index = [list(median_data.columns).index(cluster) for cluster in cluster_call]
        
        for i, gene in enumerate(other_genes):
            ggnn = current_panel + [gene]
            if corr_mapping:
                corr_matrix_df = cor_tree_mapping(map_data=map_data, median_data=median_data, genes_to_map=ggnn)
            else:
                corr_matrix_df = dist_tree_mapping(map_data=map_data, median_data=median_data, genes_to_map=ggnn)

            corr_matrix_df[corr_matrix_df.isna()] = -1
            ranked_leaf_and_value = get_top_match(corr_matrix_df)
            top_leaf = ranked_leaf_and_value['TopLeaf'].values

            if cluster_distance is None:
                match_count[i] = np.mean(cluster_call == top_leaf)
            else:
                ind_in_cluster_dist = len(median_data.columns) * (np.array([list(median_data.columns).index(x) for x in top_leaf]) - 1) + cluster_index
                match_count[i] = -np.mean(cluster_distance[ind_in_cluster_dist])
        
        wm = np.argmax(match_count)
        gene_to_add = other_genes[wm]

        if verbose:
            if optimize == "fraction_correct":
                print(f"Added {gene_to_add} with {match_count[wm]:.3f}, now matching [{len(current_panel)}].")
            else:
                print(f"Added {gene_to_add} with average cluster distance {-match_count[wm]:.3f} [{len(current_panel)}].")
        
        current_panel.append(gene_to_add)
        current_panel = build_mapping_base_marker_panel(map_data=map_data, median_data=median_data, cluster_call=cluster_call, 
                                                       panel_size=panel_size, num_subsample=num_subsample, max_fc_gene=max_fc_gene, 
                                                       qmin=qmin, seed=seed, current_panel=current_panel, 
                                                       panel_min=panel_min, verbose=verbose, corr_mapping=corr_mapping, 
                                                       optimize=optimize, cluster_distance=cluster_distance, 
                                                       cluster_genes=cluster_genes, dend=dend, percent_gene_subset=percent_gene_subset)
    
    return current_panel


def cor_tree_mapping(map_data, median_data=None, dend=None, ref_data=None, cluster_call=None, 
                     genes_to_map=None, method='pearson'):
    # Default genes_to_map to row names of map_data if not provided
    if genes_to_map is None:
        genes_to_map = map_data.index

    # If median_data is not provided
    if median_data is None:
        if cluster_call is None or ref_data is None:
            raise ValueError("Both cluster_call and ref_data must be provided if median_data is not provided.")
        # Create median_data using row-wise medians for each cluster in ref_data
        cluster_call = pd.Series(cluster_call, index=ref_data.columns)
        median_data = ref_data.groupby(cluster_call, axis=1).median()

    # If dendrogram is provided, use leaf_to_node_medians
    if dend is not None:
        # TODO: Implement leaf_to_node_medians function
        median_data = leaf_to_node_medians(dend, median_data)

    # Intersect the genes to be mapped with those in map_data and median_data
    keep_genes = list(set(genes_to_map).intersection(map_data.index).intersection(median_data.index))

    # Subset the data to include only the common genes
    map_data_subset = map_data.loc[keep_genes, :]
    median_data_subset = median_data.loc[keep_genes, :]

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = column_wise_corr_vectorized(map_data_subset.values, median_data_subset.values)
    elif method == 'spearman':
        corr_matrix = column_wise_spearman_corr_vectorized(map_data_subset.values, median_data_subset.values)
    else:
        raise ValueError("Invalid method. Please choose 'pearson' or 'spearman'.")
    corr_matrix_df = pd.DataFrame(corr_matrix, index=map_data.columns, columns=median_data.columns)
    return corr_matrix_df


def column_wise_corr_vectorized(A, B):
    # Subtract the mean from each column, ignoring NaN
    A_centered = A - np.nanmean(A, axis=0)
    B_centered = B - np.nanmean(B, axis=0)
    
    # Use masked arrays to ignore NaN values
    A_masked = np.ma.masked_invalid(A_centered)
    B_masked = np.ma.masked_invalid(B_centered)
    
    # Compute the dot product between A and B, ignoring NaN
    numerator = np.ma.dot(A_masked.T, B_masked)
    
    # Compute the denominator (standard deviations) for A and B
    A_var = np.ma.sum(A_masked ** 2, axis=0)
    B_var = np.ma.sum(B_masked ** 2, axis=0)
    
    denominator = np.sqrt(np.outer(A_var, B_var))
    
    # Calculate the correlation matrix (p x q)
    corr_matrix = numerator / denominator
    
    # Convert masked array back to regular array, filling any masked values with NaN
    return corr_matrix.filled(np.nan)


def column_wise_spearman_corr_vectorized(A, B):
    # Step 1: Rank the data, handling NaN values by ignoring them
    A_ranked = np.apply_along_axis(lambda x: rankdata(x, method='average', nan_policy='omit'), axis=0, arr=A)
    B_ranked = np.apply_along_axis(lambda x: rankdata(x, method='average', nan_policy='omit'), axis=0, arr=B)
    
    # Step 2: Compute the Pearson correlation on the ranked data using the previous vectorized Pearson method
    return column_wise_corr_vectorized(A_ranked, B_ranked)


# Placeholder for the leafToNodeMedians function
def leaf_to_node_medians(dend, median_data):
    # Implement this function based on your dendrogram logic
    return median_data


def dist_tree_mapping(dend=None, ref_data=None, map_data=None, median_data=None, 
                      cluster_call=None, genes_to_map=None, returnSimilarity=True, **kwargs):
    """
    Computes the Euclidean distance (or similarity) between map_data and median_data, 
    optionally leveraging a dendrogram structure for clustering.
    
    Parameters:
    dend (optional): Dendrogram structure, if available.
    ref_data (pd.DataFrame): Reference data matrix.
    map_data (pd.DataFrame): Data to be mapped. Defaults to ref_data.
    median_data (pd.DataFrame, optional): Precomputed medians. If None, it is computed.
    cluster_call (pd.Series): Cluster assignment of the columns in ref_data.
    genes_to_map (list, optional): List of genes to map.
    returnSimilarity (bool): Whether to return similarity instead of distance.
    **kwargs: Additional arguments for the distance function.
    
    Returns:
    pd.DataFrame: Distance matrix (or similarity matrix if returnSimilarity=True).
    """
    # If median_data is not provided, compute it based on cluster_call
    if median_data is None:
        if cluster_call is None or ref_data is None:
            raise ValueError("Both cluster_call and ref_data must be provided if median_data is not provided.")
        
        # Group by cluster_call and calculate row medians
        cluster_call = pd.Series(cluster_call, index=ref_data.columns)
        median_data = ref_data.groupby(cluster_call, axis=1).median()
        
        # Apply leafToNodeMedians (if dend is provided)
        if dend is not None:
            median_data = leaf_to_node_medians(dend, median_data)
    
    # Determine the intersection of genes to map
    if genes_to_map is None:
        genes_to_map = map_data.index
    keep_genes = list(set(genes_to_map).intersection(map_data.index).intersection(median_data.index))
    
    # If only one gene is selected, duplicate it to avoid single-dimensional data
    if len(keep_genes) == 1:
        keep_genes = keep_genes * 2
    
    # Subset map_data and median_data based on the intersected genes
    map_data_subset = map_data.loc[keep_genes, :].T  # Transposed for consistency with pdist usage
    median_data_subset = median_data.loc[keep_genes, :].T
    
    # Compute the Euclidean distance matrix
    eucDist = cdist(map_data_subset, median_data_subset, metric='euclidean', **kwargs)
    
    # Convert to a DataFrame for easier handling
    eucDist = pd.DataFrame(eucDist, index=map_data.columns, columns=median_data.columns)
    
    # If returnSimilarity is False, return the raw distance matrix
    if not returnSimilarity:
        return eucDist
    
    # If returnSimilarity is True, convert distance to similarity
    eucDist = np.sqrt(eucDist / np.max(eucDist.values))  # Normalize by max value
    similarity = 1 - eucDist
    
    return similarity


def get_top_match(memb_cl):
    # Apply the function to each row of memb_cl
    tmp_cl = np.apply_along_axis(lambda x: [memb_cl.columns[np.argmax(x)], x[np.argmax(x)]], 1, memb_cl)

    # Transpose and convert to DataFrame
    ranked_leaf_and_value = pd.DataFrame(tmp_cl, columns=["TopLeaf", "Value"])
    
    # Convert 'Value' column to numeric
    ranked_leaf_and_value["Value"] = pd.to_numeric(ranked_leaf_and_value["Value"])

    # Handle missing values
    ranked_leaf_and_value["TopLeaf"].fillna("none", inplace=True)
    ranked_leaf_and_value["Value"].fillna(0, inplace=True)

    return ranked_leaf_and_value

# Example usage
# map_data = pd.DataFrame(...)  # The data you want to map
# ref_data = pd.DataFrame(...)  # Reference data
# cluster_call = pd.Series(...)   # Cluster call for the reference data
# result = dist_tree_mapping(dend=None, ref_data=ref_data, map_data=map_data, cluster_call=cluster_call)
# print(result)




def subsample_cells(cluster_call, num_subsample=20, seed=None):
    """
    Subsamples cells from each cluster up to a maximum number (num_subsample) for each cluster.
    
    Parameters:
    cluster_call (pd.Series): A Pandas Series where the index represents cell identifiers 
                          and the values represent the cluster each cell belongs to.
    num_subsample (int): The maximum number of cells to sample from each cluster.
    seed (int): The seed for random sampling (optional, for reproducibility).
    
    Returns:
    np.ndarray: Array of sampled cell indices.
    """
    
    # List to hold the sampled cell indices
    sampled_cells = []
    
    # Group cells by cluster_call and num_subsample
    for cluster_call, cell_indices in cluster_call.groupby(cluster_call):
        if see is not None: # Set the random seed for reproducibility
            np.random.seed(seed) 
            seed += 1
        if len(cell_indices) > num_subsample:
            # Sample without replacement if the number of cells in the cluster is greater than num_subsample
            sampled_cells.extend(np.random.choice(cell_indices.index, num_subsample, replace=False))
        else:
            # Otherwise, include all cells from this cluster
            sampled_cells.extend(cell_indices.index)
    
    # Return the sampled cell indices as a NumPy array
    return np.array(sampled_cells)

# Example usage
# Assuming 'cluster_call' is a Pandas Series where index represents cell identifiers and values are cluster labels
# cluster_call = pd.Series({'cell1': 'cluster1', 'cell2': 'cluster1', 'cell3': 'cluster2', 'cell4': 'cluster2'})
# subsample_cells(cluster_call, num_subsample=1, seed=42)


# Example of how leaf_to_node_medians could be structured (to be implemented as needed)
def leaf_to_node_medians(dend, median_data):
    # Placeholder for the dendrogram-based operation on median_data
    # You need to implement this based on the logic of your dendrogram structure
    return median_data

# Example usage
# map_data = pd.DataFrame(...)  # Load or create your data
# median_data = pd.DataFrame(...)  # Load or create your data
# clusters = [...]  # Your cluster labels
# result = cor_tree_mapping(map_data, median_data, clusters=clusters)
# print(result)


############################################################################################################
### Filtering

def get_beta_score(prop_expr, return_score=True, spec_exp=2):
    """
    Calculate the beta score for each gene based on the proportion of expression values.

    Parameters:
    prop_expr (pd.DataFrame): A DataFrame of proportion of expression values.
        gene x cluster matrix.
    return_score (bool): Whether to return the beta scores or their ranks.
    spec_exp (int): Exponent for the pairwise distance calculation.

    Returns:
    np.ndarray: Array of beta scores or their ranks.
    """

    # Internal function to calculate beta score for a row
    def calc_beta(y, spec_exp=2):
        # Calculate pairwise distances
        d1 = squareform(pdist(y.reshape(-1, 1)))
        eps1 = 1e-10  # Small value to avoid division by zero
        score1 = np.sum(d1**spec_exp) / (np.sum(d1) + eps1)
        return score1
    
    # Apply calc_beta to each row of prop_expr
    beta_score = np.apply_along_axis(calc_beta, 1, prop_expr, spec_exp)
    
    # Replace NA values (NaNs) with 0
    beta_score[np.isnan(beta_score)] = 0
    
    # Return the beta scores or their ranks
    if return_score:
        return beta_score
    else:
        score_rank = np.argsort(-beta_score)  # Rank in descending order
        return score_rank


def filter_panel_genes(summary_expr, prop_expr=None, on_clusters=None, off_clusters=None, 
                       gene_lengths=None, starting_genes=None, num_binary_genes=500, 
                       min_on=10, max_on=250, max_off=50, min_length=960, 
                       fraction_on_clusters=0.5, on_threshold=0.5, 
                       exclude_genes=None, exclude_families=None):
    """
    Filters genes based on expression and other criteria.

    Parameters:
    summary_expr (pd.DataFrame): A DataFrame of gene expression values, usually a median.
        gene x cluster matrix.
    prop_expr (pd.DataFrame): A DataFrame of proportion of expression values.
        gene x cluster matrix.
    on_clusters (list): List of cluster names or indices to consider as 'on' clusters.
    off_clusters (list): List of cluster names or indices to consider as 'off' clusters.
    gene_lengths (np.ndarray): Array of gene lengths.
    starting_genes (list): List of genes to start with.
    num_binary_genes (int): Number of binary genes to select.
    min_on (int): Minimum expression value for max 'on' clusters.
    max_on (int): Maximum expression value for max 'on' clusters.
    max_off (int): Maximum expression value for max 'off' clusters.
    min_length (int): Minimum gene length.
    fraction_on_clusters (float): Fraction of 'on' clusters that a gene should be expressed in.
    on_threshold (float): Threshold for 'on' expression.
    exclude_genes (list): List of genes to exclude.
    exclude_families (list): List of gene families to exclude.

    Returns:
    list: List of genes that pass the filtering criteria.
    """
    
    if starting_genes is None:
        starting_genes = ["GAD1", "SLC17A7"]
    
    if exclude_families is None:
        exclude_families = ["LOC", "LINC", "FAM", "ORF", "KIAA", "FLJ", "DKFZ", "RIK", "RPS", "RPL", "\\-"]

    # Check if summary_expr is a matrix (or DataFrame)
    if not isinstance(summary_expr, (np.ndarray, pd.DataFrame)):
        raise ValueError("summaryExpr must be a matrix or DataFrame of numeric values.")
    
    if not np.issubdtype(summary_expr.values[0, 0], np.number):
        raise ValueError("summaryExpr must contain numeric values.")
    
    if summary_expr.index is None:
        raise ValueError("Please provide summaryExpr with genes as row names.")

    if not isinstance(fraction_on_clusters, (int, float)):
        raise ValueError("fractionOnClusters needs to be numeric.")
    
    # If franction_on_clusters is greater than 1, assume it is in % and convert to fraction
    if fraction_on_clusters > 1:
        fraction_on_clusters /= 100
    
    genes = summary_expr.index
    genes_u = genes.str.upper()
    exclude_families = [ef.upper() for ef in exclude_families]
    
    # Create a boolean array for excluded genes and families
    exclude_genes = np.isin(genes, exclude_genes)    
    for ef in exclude_families:
        exclude_genes |= genes_u.str.contains(ef)

    # Handle on_clusters and off_clusters
    if isinstance(on_clusters, list) and all(isinstance(x, str) for x in on_clusters):
        on_clusters = np.isin(summary_expr.columns, on_clusters)
    elif isinstance(on_clusters, list) and all(isinstance(x, int) for x in on_clusters):
        on_clusters = np.isin(range(summary_expr.shape[1]), on_clusters)

    if np.sum(on_clusters) < 2:
        raise ValueError("Please provide at least two onClusters.")
    
    if off_clusters is not None:
        if isinstance(off_clusters, list) and all(isinstance(x, str) for x in off_clusters):
            off_clusters = np.isin(summary_expr.columns, off_clusters)
        elif isinstance(off_clusters, list) and all(isinstance(x, int) for x in off_clusters):
            off_clusters = np.isin(range(summary_expr.shape[1]), off_clusters)

    # Calculate max expression for on and off clusters
    max_expr_on = summary_expr.loc[:, on_clusters].max(axis=1)
    
    if off_clusters is not None:
        if np.sum(off_clusters) > 1:
            max_expr_off = summary_expr.loc[:, off_clusters].max(axis=1)
        elif np.sum(off_clusters) == 1:
            max_expr_off = summary_expr.loc[:, off_clusters]
    else:
        max_expr_off = np.full_like(max_expr_on, -np.inf)

    # Gene length validation
    if gene_lengths is not None:
        if len(gene_lengths) != len(summary_expr):
            raise ValueError("geneLengths must be of the same length as the rows of summaryExpr.")
        if not isinstance(gene_lengths, (np.ndarray, list)):
            raise ValueError("geneLengths must be numeric.")
    else:
        gene_lengths = np.full_like(max_expr_on, np.inf)

    # Filter genes
    keep_genes = (~exclude_genes) & (max_expr_on > min_on) & (max_expr_on <= max_on) & \
                 (max_expr_off <= max_off) & (gene_lengths >= min_length) & \
                 (prop_expr.loc[:, on_clusters].gt(on_threshold).mean(axis=1) <= fraction_on_clusters) & \
                 (prop_expr.loc[:, on_clusters].gt(on_threshold).mean(axis=1) > 0)
    
    keep_genes = np.nan_to_num(keep_genes, nan=False).astype(bool)

    print(f"{np.sum(keep_genes)} total genes pass constraints prior to binary score calculation.")

    # If fewer genes pass constraints than numBinaryGenes
    if np.sum(keep_genes) <= num_binary_genes:
        print(f"Warning: Fewer genes pass constraints than {num_binary_genes}, so binary score was not calculated.")
        return sorted(list(set(genes[keep_genes]).union(starting_genes)))

    # Calculate beta score (rank)
    top_beta = get_beta_score(prop_expr.loc[keep_genes, on_clusters], False)
    
    run_genes = genes[keep_genes][top_beta < num_binary_genes]
    run_genes = sorted(list(set(run_genes).union(starting_genes)))
    
    return run_genes
