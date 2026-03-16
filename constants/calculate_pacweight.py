import os
import json
import numpy as np
from sklearn.decomposition import PCA


METRIC_KEYS = [
    'average_local_rotation_variance',
    'average_local_translation_variance',
    'average_global_rotation_variance',
    'average_global_translation_variance',
    'average_depth_consistency_error',
]

RAW_METRIC_SHORT_NAMES = ["LRotVar", "LTransVar", "GRotVar", "GTransVar", "DepthCons"]
NORM_METRIC_SHORT_NAMES = ["NormLRotV", "NormLTraV", "NormGRotV", "NormGTraV", "NormDepCo"]
ZSCORE_METRIC_SHORT_NAMES = ["ZRotVar", "ZTraVar", "ZGRotV", "ZGTraV", "ZDepCon"]

def process_one_folder(folder_path):
    """
    Load all .json files in folder_path, extract metrics (treating null as NaN),
    return their nan-means and the count of JSON files.
    """
    metrics = {key: [] for key in METRIC_KEYS}
    json_count = 0
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.json'):
            continue
        json_count += 1
        with open(os.path.join(folder_path, fname), 'r') as f:
            data = json.load(f)
        for key in metrics:
            val = data.get(key, None)
            metrics[key].append(np.nan if val is None else float(val))

    averages = {}
    for key, lst in metrics.items():
        arr = np.array(lst, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            averages[key] = np.nan
        else:
            averages[key] = np.nanmean(arr)
    return averages, json_count

def collect_all_dataset_raw_metrics(base_directory):
    """
    Collects raw average metrics from all subfolders in the base_directory.
    """
    all_metrics_list_of_dicts = []
    dataset_info_list = []

    if not os.path.isdir(base_directory):
        print(f"Error: Root directory '{base_directory}' not found.")
        return all_metrics_list_of_dicts, dataset_info_list

    for name in sorted(os.listdir(base_directory)):
        subdir = os.path.join(base_directory, name)
        if not os.path.isdir(subdir):
            continue

        averages, count = process_one_folder(subdir)

        dataset_metrics_dict = {'dataset_name': name}
        for key in METRIC_KEYS:
            dataset_metrics_dict[key] = averages.get(key, np.nan)
        all_metrics_list_of_dicts.append(dataset_metrics_dict)

        dataset_info_list.append({
            'name': name,
            'json_count': count,
            'raw_averages': averages
        })

    return all_metrics_list_of_dicts, dataset_info_list

LOG_TRANSFORM_METRICS = [
    'average_local_rotation_variance',
    'average_local_translation_variance',
    'average_global_rotation_variance',
    'average_global_translation_variance'
]
def normalize_metrics_z_then_minmax(all_metrics_list_of_dicts, metric_keys_list):
    """
    Normalizes metrics using an optional Log-Transform, followed by Z-score standardization,
    and finally Min-Max scaling on Z-scores.
    """
    num_datasets = len(all_metrics_list_of_dicts)
    normalization_params_log = {}

    if num_datasets == 0:
        for key in metric_keys_list:
            normalization_params_log[key] = {
                'mu_raw': np.nan, 'sigma_raw': np.nan,
                'z_score_min': np.nan, 'z_score_max': np.nan
            }
        return (np.array([]).reshape(0,len(metric_keys_list)),
                np.array([]).reshape(0,len(metric_keys_list)),
                normalization_params_log)

    num_metrics = len(metric_keys_list)
    raw_metrics_array = np.full((num_datasets, num_metrics), np.nan)

    for i, data_dict in enumerate(all_metrics_list_of_dicts):
        for j, key in enumerate(metric_keys_list):
            raw_metrics_array[i, j] = data_dict.get(key, np.nan)

    z_scores_array = np.full_like(raw_metrics_array, np.nan)
    m_double_prime_array = np.full_like(raw_metrics_array, np.nan)

    for j, key in enumerate(metric_keys_list):
        raw_metric_column = raw_metrics_array[:, j]


        if key in LOG_TRANSFORM_METRICS:

            safe_column = np.where(raw_metric_column < 0, 0, raw_metric_column)
            processed_column = np.log1p(safe_column)
        else:

            processed_column = raw_metric_column

        current_metric_params = {
            'mu_raw': np.nan, 'sigma_raw': np.nan,
            'z_score_min': np.nan, 'z_score_max': np.nan
        }

        if np.all(np.isnan(processed_column)):
            normalization_params_log[key] = current_metric_params
            continue


        mu_k = np.nanmean(processed_column)
        sigma_k = np.nanstd(processed_column)

        current_metric_params['mu_raw'] = mu_k if not np.isnan(mu_k) else None
        current_metric_params['sigma_raw'] = sigma_k if not np.isnan(sigma_k) else None

        if np.isnan(mu_k):
            normalization_params_log[key] = current_metric_params
            continue

        if sigma_k == 0 or np.isnan(sigma_k):
            z_scores_array[:, j] = np.where(np.isnan(processed_column), np.nan, 0.0)
        else:

            z_scores_array[:, j] = (processed_column - mu_k) / sigma_k

        current_z_scores_col = z_scores_array[:, j]

        if np.all(np.isnan(current_z_scores_col)):
            normalization_params_log[key] = current_metric_params
            continue

        z_min_k = np.nanmin(current_z_scores_col)
        z_max_k = np.nanmax(current_z_scores_col)
        current_metric_params['z_score_min'] = z_min_k if not np.isnan(z_min_k) else None
        current_metric_params['z_score_max'] = z_max_k if not np.isnan(z_max_k) else None

        if np.isnan(z_min_k):
            normalization_params_log[key] = current_metric_params
            continue

        denominator = z_max_k - z_min_k
        if denominator == 0:
            m_double_prime_array[:, j] = np.where(np.isnan(current_z_scores_col), np.nan, 0.0)
        else:
            m_double_prime_array[:, j] = (current_z_scores_col - z_min_k) / denominator

        normalization_params_log[key] = current_metric_params

    return m_double_prime_array, z_scores_array, normalization_params_log

def calculate_pca_derived_weights(m_norm_array, metric_keys_list):
    num_datasets, num_total_metrics = m_norm_array.shape
    if num_datasets < 2:
        print("Warning: PCA requires at least 2 datasets. Falling back to equal weights.")
        return {key: 1.0 / num_total_metrics for key in metric_keys_list}
    valid_metric_indices = [j for j in range(num_total_metrics) if not np.isnan(m_norm_array[:, j]).all()]
    if not valid_metric_indices:
        print("ERROR: All metric columns are entirely NaN. Cannot perform PCA. Falling back to equal weights for all metrics.")
        return {key: 1.0 / num_total_metrics for key in metric_keys_list}
    if len(valid_metric_indices) < num_total_metrics:
        invalid_metric_names = [metric_keys_list[j] for j in range(num_total_metrics) if j not in valid_metric_indices]
        print(f"Info: The following metrics are all NaNs and will be excluded from PCA, assigned zero weight: {invalid_metric_names}")
    m_pca_input = m_norm_array[:, valid_metric_indices]
    col_means_for_imputation = np.nanmean(m_pca_input, axis=0)
    for j_pca_col_idx in range(m_pca_input.shape[1]):
        column_data = m_pca_input[:, j_pca_col_idx]
        nan_mask = np.isnan(column_data)
        if np.any(nan_mask):
             m_pca_input[nan_mask, j_pca_col_idx] = col_means_for_imputation[j_pca_col_idx]
    std_devs = np.std(m_pca_input, axis=0)
    if (std_devs == 0).any():
        zero_std_cols_in_pca_input_indices = np.where(std_devs == 0)[0]
        original_indices_of_zero_std = [valid_metric_indices[i] for i in zero_std_cols_in_pca_input_indices]
        zero_std_names = [metric_keys_list[i] for i in original_indices_of_zero_std]
        print(f"Info: After imputation, the following metrics have zero variance: {zero_std_names}. Their PCA loadings will likely be zero.")
    pc1_loadings_for_valid_metrics = np.zeros(len(valid_metric_indices))
    explained_variance_ratio_pc1 = 0.0
    if m_pca_input.shape[1] == 0:
        print("ERROR: No valid metrics for PCA. Falling back to equal weights.")
        return {key: 1.0 / num_total_metrics for key in metric_keys_list}
    elif m_pca_input.shape[1] == 1:
        print("Info: Only one valid metric for PCA. Assigning full weight to it.")
        pc1_loadings_for_valid_metrics = np.array([1.0])
        explained_variance_ratio_pc1 = 1.0
    else:
        n_comp = min(m_pca_input.shape[0], m_pca_input.shape[1])
        pca = PCA(n_components=n_comp)
        try:
            pca.fit(m_pca_input)
            pc1_loadings_for_valid_metrics = pca.components_[0]
            explained_variance_ratio_pc1 = pca.explained_variance_ratio_[0]
            print(f"Info: PC1 explained variance ratio (among {len(valid_metric_indices)} valid metrics): {explained_variance_ratio_pc1:.4f}")
        except ValueError as e:
            print(f"ERROR during PCA fitting: {e}. Falling back to equal weights among valid metrics.")
            final_weights_values_fallback = np.zeros(num_total_metrics)
            if len(valid_metric_indices) > 0:
                equal_weight_for_valid = 1.0 / len(valid_metric_indices)
                for original_idx in valid_metric_indices:
                    final_weights_values_fallback[original_idx] = equal_weight_for_valid
            else:
                equal_weight_for_valid = 1.0 / num_total_metrics
                for i in range(num_total_metrics):
                    final_weights_values_fallback[i] = equal_weight_for_valid
            return {key: weight for key, weight in zip(metric_keys_list, final_weights_values_fallback)}
    abs_loadings_for_valid_metrics = np.abs(pc1_loadings_for_valid_metrics)
    sum_abs_loadings = np.sum(abs_loadings_for_valid_metrics)
    final_weights_values = np.zeros(num_total_metrics)
    if sum_abs_loadings < 1e-9:
        print("Warning: Sum of absolute PC1 loadings is near zero. Falling back to equal weights among valid metrics.")
        if len(valid_metric_indices) > 0:
            equal_weight_for_valid = 1.0 / len(valid_metric_indices)
            for original_idx in valid_metric_indices:
                final_weights_values[original_idx] = equal_weight_for_valid
    else:
        normalized_loadings_for_valid_metrics = abs_loadings_for_valid_metrics / sum_abs_loadings
        for i, original_idx in enumerate(valid_metric_indices):
            final_weights_values[original_idx] = normalized_loadings_for_valid_metrics[i]
    pca_derived_weights = {key: weight for key, weight in zip(metric_keys_list, final_weights_values)}
    print("Info: Successfully derived PCA weights.")
    return pca_derived_weights

def calculate_composite_metric_scores_dynamic_weights(m_double_prime_array, all_metrics_list_of_dicts, current_weights_dict, metric_keys_list_arg):
    composite_scores_details = []
    num_datasets = m_double_prime_array.shape[0]
    for i in range(num_datasets):
        dataset_name = all_metrics_list_of_dicts[i]['dataset_name']
        sgc_score_numerator = 0.0
        total_weight_for_non_nan = 0.0
        all_metrics_are_nan_for_score = True
        current_m_double_prime_dict = {}
        for j, key in enumerate(metric_keys_list_arg):
            m_double_prime_val = m_double_prime_array[i, j]
            current_m_double_prime_dict[key] = m_double_prime_val
            if not np.isnan(m_double_prime_val):
                sgc_score_numerator += current_weights_dict[key] * m_double_prime_val
                total_weight_for_non_nan += current_weights_dict[key]
                all_metrics_are_nan_for_score = False
        if all_metrics_are_nan_for_score or total_weight_for_non_nan == 0:
            final_score_for_dataset = np.nan
        else:
            final_score_for_dataset = sgc_score_numerator / total_weight_for_non_nan
        composite_scores_details.append({
            'dataset_name': dataset_name,
            'SGC_score': final_score_for_dataset,
            'normalized_metrics_M_double_prime': current_m_double_prime_dict
        })
    composite_scores_details.sort(key=lambda x: (np.isnan(x['SGC_score']), x['SGC_score']))
    return composite_scores_details


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    root_directory = os.path.join(script_dir, 'constants')

    if not os.path.exists(root_directory):
         print(f"WARNING: The results directory does not exist: {root_directory}")
         print("Please ensure the 'root_directory' points to your data. No data will be processed.")

    all_raw_metrics_data, dataset_original_info_list = collect_all_dataset_raw_metrics(root_directory)

    if not all_raw_metrics_data:
        print(f"No datasets found or processed in '{root_directory}'. Please check the path.")
        return

    m_double_prime_array, z_scores_array, normalization_parameters = normalize_metrics_z_then_minmax(all_raw_metrics_data, METRIC_KEYS)

    z_score_map = {}
    for i, data_dict in enumerate(all_raw_metrics_data):
        dataset_name = data_dict['dataset_name']
        z_scores_for_dataset = {}
        for j, key in enumerate(METRIC_KEYS):
            z_scores_for_dataset[key] = z_scores_array[i, j]
        z_score_map[dataset_name] = z_scores_for_dataset

    print("\n--- Calculating PCA-Derived Weights ---")
    pca_derived_weights = calculate_pca_derived_weights(m_double_prime_array, METRIC_KEYS)
    print("PCA-Derived Weights:")
    for key, weight in pca_derived_weights.items():
        print(f"  {key}: {weight:.4f}")

    ranked_datasets_details = calculate_composite_metric_scores_dynamic_weights(
        m_double_prime_array,
        all_raw_metrics_data,
        pca_derived_weights,
        METRIC_KEYS
    )

    dataset_info_map = {info['name']: info for info in dataset_original_info_list}
    header_parts = [f"{'Rank':<5}", f"{'Dataset':<25}", f"{'SGC Score':<15}", f"{'JSONs':<7}"]
    for name in RAW_METRIC_SHORT_NAMES:
        header_parts.append(f"{name:<10}")
    for name in ZSCORE_METRIC_SHORT_NAMES:
        header_parts.append(f"{name:<10}")
    for name in NORM_METRIC_SHORT_NAMES:
        header_parts.append(f"{name:<10}")
    header = " | ".join(header_parts)
    output_lines = [header, "-" * len(header)]

    for rank, details in enumerate(ranked_datasets_details):
        dataset_name = details['dataset_name']
        sgc = details['SGC_score']
        norm_metrics_m_double_prime = details['normalized_metrics_M_double_prime']
        original_info = dataset_info_map.get(dataset_name)
        raw_avgs = original_info['raw_averages'] if original_info else {key: np.nan for key in METRIC_KEYS}
        json_count = original_info['json_count'] if original_info else "N/A"

        z_scores = z_score_map.get(dataset_name, {key: np.nan for key in METRIC_KEYS})
        rank_str = f"{rank + 1}" if not np.isnan(sgc) else "N/A"
        sgc_str = f"{sgc:.4f}" if not np.isnan(sgc) else "N/A"
        json_count_str = f"{json_count}"
        line_parts = [f"{rank_str:<5}", f"{dataset_name:<25}", f"{sgc_str:<15}", f"{json_count_str:<7}"]

        for key in METRIC_KEYS:
            val = raw_avgs.get(key, np.nan)
            line_parts.append(f"{val:<10.4f}" if not np.isnan(val) else f"{'N/A':<10}")

        for key in METRIC_KEYS:
            val = z_scores.get(key, np.nan)
            line_parts.append(f"{val:<10.4f}" if not np.isnan(val) else f"{'N/A':<10}")

        for key in METRIC_KEYS:
            val = norm_metrics_m_double_prime.get(key, np.nan)
            line_parts.append(f"{val:<10.3f}" if not np.isnan(val) else f"{'N/A':<10}")
        output_lines.append(" | ".join(line_parts))

    print("\n--- 3D Consistency Benchmark Results (using PCA-Derived Weights) ---")
    for line in output_lines:
        print(line)

    summary_file_path = os.path.join(root_directory, "composite_3d_consistency_summary_pca_weights.txt")

    try:
        with open(summary_file_path, 'w') as f:
            f.write("Composite 3D Consistency Score (SGC): Lower is better.\n")
            f.write("Normalization: Z-score standardization, then Min-Max scaling of Z-scores to [0 (best), 1 (worst)].\n")

            f.write("\n--- Normalization Parameters Used (calculated from the processed datasets) ---\n")

            def convert_for_json(obj):
                if isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_for_json(i) for i in obj]
                if isinstance(obj, np.floating):
                    return float(obj) if not np.isnan(obj) else None
                if isinstance(obj, np.integer):
                    return int(obj)
                if obj is np.nan:
                    return None
                return obj

            serializable_norm_params = convert_for_json(normalization_parameters)
            f.write(json.dumps(serializable_norm_params, indent=2))
            f.write("\n\n")

            f.write("--- PCA-Derived Weights ---\n")
            f.write(f"Calculated PCA Weights: {json.dumps(pca_derived_weights, indent=2)}\n\n")

            f.write("--- Detailed Results Table ---\n")
            for line in output_lines:
                f.write(line + "\n")
        print(f"\nSummary report written to: {summary_file_path}")
    except IOError as e:
        print(f"\nError writing summary file: {e}")

if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        print(f"Warning: '__file__' not defined, using current working directory for 'script_dir': {script_dir}")

    main_root_directory = os.path.join(script_dir, '3d_consistency', 'result','fast_cca')

    if not os.path.exists(main_root_directory):
         print(f"WARNING (from __main__): The default results directory for main() does not exist: {main_root_directory}")
         print("Please ensure the 'root_directory' in the main() function points to your data.")

    main()