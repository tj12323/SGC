#!/usr/bin/env bash

BASE_NAME="mygenvideo/k400-val"
MAIN_BASE_DIR="data/${BASE_NAME}"
RESULT_BASE_DIR="third-party/SegAnyMo/result/sam2"

GPUS=(1 3 5 6 7)
NUM_GPUS=${#GPUS[@]}
active_jobs=0 
gpu_idx=0     

for parent_dir in "${MAIN_BASE_DIR}"/*; do
    if [ -d "${parent_dir}" ]; then
        parent_name=$(basename "${parent_dir}")
        echo "Processing top-level directory: ${parent_name}"

        FRAMES_BASE="${parent_dir}/resize_images"
        DEPTH_BASE="${parent_dir}/video_depth_anything"
        TRACKS_BASE="${parent_dir}/delta"
        MOS_BASE="${RESULT_BASE_DIR}/${parent_name}/initial_preds"

        if [ -d "${FRAMES_BASE}" ]; then
            for video_dir_path in "${FRAMES_BASE}"/*; do
                if [ -d "${video_dir_path}" ]; then
                    video_name=$(basename "${video_dir_path}")
                    if [ -d "${DEPTH_BASE}" ] && [ -d "${TRACKS_BASE}" ]; then
                        if [ "$active_jobs" -ge "$NUM_GPUS" ]; then
                            echo "Reached max parallel jobs ($NUM_GPUS). Waiting for a slot..."
                            wait -n 
                            active_jobs=$((active_jobs - 1))
                        fi

                        current_gpu_id=${GPUS[$gpu_idx]}
                        
                        echo "  Launching task for video: ${video_name} (from ${parent_name}) on GPU ${current_gpu_id}"

                        (
                            echo "  [GPU ${current_gpu_id}] Processing video: ${video_name} from ${parent_name}"
                            echo "    [GPU ${current_gpu_id}]   Frames base: ${FRAMES_BASE}"
                            echo "    [GPU ${current_gpu_id}]   Depth base: ${DEPTH_BASE}"
                            echo "    [GPU ${current_gpu_id}]   MOS base: ${MOS_BASE}"
                            echo "    [GPU ${current_gpu_id}]   Tracks base: ${TRACKS_BASE}"

                            CUDA_VISIBLE_DEVICES=${current_gpu_id} python sgc/calculate_glo_fast.py \
                                --frames_dir "${FRAMES_BASE}" \
                                --depth_dir "${DEPTH_BASE}" \
                                --mos_dir "${MOS_BASE}" \
                                --tracks_dir "${TRACKS_BASE}" \
                                --video_name "${video_name}" \
                                --n_depth_clusters 10
                                # --segmentation_method "grid"
                            
                            local exit_status=$? # 捕获python脚本的退出状态
                            if [ $exit_status -ne 0 ]; then
                                echo "  [GPU ${current_gpu_id}] ERROR: Task for ${video_name} failed with status ${exit_status}."
                            else
                                echo "  [GPU ${current_gpu_id}] Finished ${video_name}"
                            fi
                            echo "  [GPU ${current_gpu_id}] --------------------------"
                        ) &

                        active_jobs=$((active_jobs + 1))
                        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS )) # 循环到下一个GPU

                    else
                        echo "  Warning: Some base directories for video '${video_name}' in '${parent_name}' do not exist. Skipping."
                        echo "    Expected Depth Base: ${DEPTH_BASE} (exists: $([ -d "${DEPTH_BASE}" ] && echo true || echo false))"
                        echo "    Expected MOS Base: ${MOS_BASE} (exists: $([ -d "${MOS_BASE}" ] && echo true || echo false))"
                        echo "    Expected Tracks Base: ${TRACKS_BASE} (exists: $([ -d "${TRACKS_BASE}" ] && echo true || echo false))"
                        echo "  --------------------------"
                    fi
                fi
            done
        else
            echo "  Warning: Frames base directory '${FRAMES_BASE}' not found in '${parent_name}'. Skipping."
            echo "  --------------------------"
        fi
    fi
done

# 等待所有剩余的后台任务完成
echo "Waiting for all remaining jobs to finish..."
wait
echo "All jobs completed."
echo "All top-level directories processed."