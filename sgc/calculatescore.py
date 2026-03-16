import json
import re
import numpy as np
import os

METRIC_KEYS = [
    'average_local_rotation_variance',
    'average_local_translation_variance',
    'average_global_rotation_variance',
    'average_global_translation_variance',
    'average_depth_consistency_error',
]

ABBREV_KEYS = {
    'average_local_rotation_variance': 'LocRot',
    'average_local_translation_variance': 'LocTra',
    'average_reprojection_error': 'Reproj',
    'average_global_rotation_variance': 'GloRot',
    'average_global_translation_variance': 'GloTra',
    'average_depth_consistency_error': 'Depth',
}

LOG_TRANSFORM_METRICS = [
    'average_local_rotation_variance',
    'average_local_translation_variance',
    'average_global_rotation_variance',
    'average_global_translation_variance'
]

def parse_summary_file(summary_path):
    with open(summary_path, 'r') as f:
        text = f.read()

    norm_match = re.search(
        r"--- Normalization Parameters Used .*?---\n\s*(\{.*?\})\n\n",
        text, flags=re.S
    )
    weight_match = re.search(
        r"--- PCA-Derived Weights .*?Calculated PCA Weights:\s*(\{.*?\})\n",
        text, flags=re.S
    )
    if not norm_match or not weight_match:
        raise ValueError("The required JSON block cannot be found in the summary file")

    return json.loads(norm_match.group(1)), json.loads(weight_match.group(1))

def compute_video_score(video_json_path, norm_params, pca_weights):
    with open(video_json_path, 'r') as f:
        raw = json.load(f)

    mpp = {}
    raw_vals = {}
    processed_vals = {}
    z_scores = {}
    num = 0.0
    denom = 0.0

    for key in METRIC_KEYS:
        val = raw.get(key, None)
        raw_vals[key] = val if val is not None else np.nan

        params = norm_params.get(key, {})
        μ = params.get('mu_raw', None)
        σ = params.get('sigma_raw', None)
        zmin = params.get('z_score_min', None)
        zmax = params.get('z_score_max', None)
        w = pca_weights.get(key, 0.0)

        if val is None or μ is None or σ is None or zmin is None or zmax is None:
            mpp[key] = np.nan
            processed_vals[key] = np.nan
            z_scores[key] = np.nan
            continue

        if key in LOG_TRANSFORM_METRICS:
            safe_val = max(0, val)
            processed_val = np.log1p(safe_val)
        else:
            processed_val = val

        processed_vals[key] = processed_val

        if σ == 0:
            z = 0.0
        else:
            z = (processed_val - μ) / σ
        z_scores[key] = z

        span = zmax - zmin
        if span == 0:
            m_val = 0.0
        else:
            m_val = (z - zmin) / span
            m_val = np.clip(m_val, 0.0, None)
        mpp[key] = m_val

        if not np.isnan(m_val) and w > 0:
            num += w * m_val
            denom += w

    SGC = num / denom if denom > 0 else np.nan

    return {
        'normalized_metrics_m_double_prime': mpp,
        'raw_metrics': raw_vals,
        'processed_metrics': processed_vals,
        'z_scores': z_scores,
        'SGC_score': SGC
    }

def main():
    root_directory = "result"
    summary_txt_path = "constants/composite_4d_consistency_summary_pca_weights.txt"

    if not os.path.isdir(root_directory):
        print(f"Error: The root directory does not exist:{root_directory}")
        return
    if not os.path.isfile(summary_txt_path):
        print(f"Error: Cannot find the summary file:{summary_txt_path}")
        return


    print("Parsing the summary file...")
    norm_params, pca_weights = parse_summary_file(summary_txt_path)

    print(f"\nNumber of loaded metrics: {len(METRIC_KEYS)}")
    print(f"Normalization parameters: {list(norm_params.keys())}")
    print(f"PCA weights: {list(pca_weights.keys())}\n")


    folder_avg_scores = []


    for name in sorted(os.listdir(root_directory)):
        subdir = os.path.join(root_directory, name)
        if not os.path.isdir(subdir):
            continue

        json_files = [f for f in sorted(os.listdir(subdir)) if f.lower().endswith('.json')]
        out_txt = os.path.join(root_directory, f"{name}_score_detailed.txt")

        results = []
        errors = []

        with open(out_txt, 'w', encoding='utf-8') as fout:
            fout.write(f"Folder: {name}\n")
            fout.write("=" * 160 + "\n")
            fout.write(f"Number of JSON files: {len(json_files)}\n\n")


            header_parts = [f"{'Filename':<35}", f"{'SGC':<10}"]

            for k in METRIC_KEYS:
                header_parts.append(f"{ABBREV_KEYS[k]}_Raw")
            for k in METRIC_KEYS:
                header_parts.append(f"{ABBREV_KEYS[k]}_Z")
            for k in METRIC_KEYS:
                header_parts.append(f"{ABBREV_KEYS[k]}_Norm")

            header = " | ".join(header_parts)
            fout.write(header + "\n")
            fout.write("=" * len(header) + "\n")


            for fname in json_files:
                path_json = os.path.join(subdir, fname)
                try:
                    res = compute_video_score(path_json, norm_params, pca_weights)
                    results.append((fname, res))
                except Exception as e:
                    errors.append((fname, str(e)))


            results_sorted = sorted(
                results,
                key=lambda x: (np.isnan(x[1]['SGC_score']), x[1]['SGC_score'])
            )


            valid_scores = []
            valid_raw = {k: [] for k in METRIC_KEYS}
            valid_z = {k: [] for k in METRIC_KEYS}
            valid_norm = {k: [] for k in METRIC_KEYS}


            for fname, res in results_sorted:
                score = res['SGC_score']
                raw_vals = res['raw_metrics']
                z_vals = res['z_scores']
                norm_vals = res['normalized_metrics_m_double_prime']

                score_str = f"{score:.6f}" if not np.isnan(score) else "N/A"
                if not np.isnan(score):
                    valid_scores.append(score)

                line_parts = [f"{fname:<35}", f"{score_str:<10}"]


                for k in METRIC_KEYS:
                    v = raw_vals.get(k, np.nan)
                    if not np.isnan(v):
                        valid_raw[k].append(v)
                        line_parts.append(f"{v:<12.6f}")
                    else:
                        line_parts.append(f"{'N/A':<12}")


                for k in METRIC_KEYS:
                    v = z_vals.get(k, np.nan)
                    if not np.isnan(v):
                        valid_z[k].append(v)
                        line_parts.append(f"{v:<10.4f}")
                    else:
                        line_parts.append(f"{'N/A':<10}")


                for k in METRIC_KEYS:
                    v = norm_vals.get(k, np.nan)
                    if not np.isnan(v):
                        valid_norm[k].append(v)
                        line_parts.append(f"{v:<10.4f}")
                    else:
                        line_parts.append(f"{'N/A':<10}")

                fout.write(" | ".join(line_parts) + "\n")


            if errors:
                fout.write("\n" + "=" * 80 + "\n")
                fout.write("--- Failures ---\n")
                for fname, err in errors:
                    fout.write(f"{fname:<35} Error: {err}\n")


            fout.write("\n" + "=" * 80 + "\n")
            fout.write("Statistics Summary\n")
            fout.write("=" * 80 + "\n")

            if valid_scores:
                avg_score = np.mean(valid_scores)
                fout.write(f"Average SGC score: {avg_score:.6f}\n")
                fout.write(f"Median SGC score: {np.median(valid_scores):.6f}\n")
                fout.write(f"Minimum SGC score: {np.min(valid_scores):.6f}\n")
                fout.write(f"Maximum SGC score: {np.max(valid_scores):.6f}\n\n")


                folder_avg_scores.append((name, avg_score))

                fout.write("Average of the original indicators:\n")
                for k in METRIC_KEYS:
                    if valid_raw[k]:
                        fout.write(f"  {ABBREV_KEYS[k]:<10}: {np.mean(valid_raw[k]):.6f}\n")
            else:
                fout.write("There is no valid data available for analysis.\n")

        print(f"✓ Wrote detailed scores for '{name}' -> {out_txt}")


    if folder_avg_scores:
        summary_output = os.path.join(root_directory, "all_folders_avg_sgc_summary.txt")
        folder_avg_scores.sort(key=lambda x: x[1])

        with open(summary_output, 'w', encoding='utf-8') as fout:
            fout.write("Statistics on the average SGC scores for all subfolders\n")
            fout.write("=" * 60 + "\n")
            fout.write(f"{'Folder Name':<40} {'Average SGC score':<15}\n")
            fout.write("=" * 60 + "\n")

            for folder_name, avg_score in folder_avg_scores:
                fout.write(f"{folder_name:<40} {avg_score:<15.6f}\n")

            fout.write("\n" + "=" * 60 + "\n")
            fout.write("Overall Statistics:\n")
            all_avgs = [score for _, score in folder_avg_scores]
            fout.write(f"Number of folders: {len(all_avgs)}\n")
            fout.write(f"Overall average score: {np.mean(all_avgs):.6f}\n")
            fout.write(f"Median score: {np.median(all_avgs):.6f}\n")
            fout.write(f"Lowest average score: {np.min(all_avgs):.6f} ({folder_avg_scores[0][0]})\n")
            fout.write(f"Highest average score: {np.max(all_avgs):.6f} ({folder_avg_scores[-1][0]})\n")

        print(f"\n✓ Average scores for all folders have been calculated -> {summary_output}")
    else:
        print("\n⚠ No valid subfolder data was found.")

if __name__ == '__main__':
    main()
