
#!/bin/bash

# Create transcripts directory if it doesn't exist
mkdir -p ./transcripts

# Array of all video files to process
videos=(
    # Q5bvideos
    "Q5videos/Q5b_Video_3b23e96a-1f6e-4490-bbe2-206b0884d3f8_e58eb9d3-21ac-46ee-b95a-d5a7c3ca21a0.mp4"
    "Q5videos/Q5b_Video_8b8121e7-52a5-49d5-8deb-6445f46d193d_901c9b22-3022-4251-bcae-6a351700d0a4.mp4"
    "Q5videos/Q5b_Video_68f8e9db-3af0-4d43-9a3d-3cf9e0869d97_742901b1-6f0f-479e-b60c-c72a9f975d90.mp4"
    "Q5videos/Q5b_Video_618e0016-90c7-4e7e-a390-b848e0ef3ea6_5f7cae78-9f56-4c8a-bd11-664a252a5b4b.mp4"
    "Q5videos/Q5b_Video_8642b6ca-a3bf-4a74-8f18-ff5aee5589cd_8fbb9a61-71a2-4853-bd69-2514d4c0bb9e.mp4"
    "Q5videos/Q5b_Video_a154eae9-1906-4c2a-b836-ea94708b3eaa_2722e930-992c-4df7-8a63-ef01e7c8b7a0.mp4"
    "Q5videos/Q5b_Video_e7021045-c488-44be-a32d-494f59531f93_a759b156-47c9-4503-96d8-349d43eb49e1.mp4"
    "Q5videos/Q5b_Video_f3e6c8bb-8e2e-4c33-b95f-e92aa7411189_384e6b12-207e-41a5-97f3-26871c21b02c.mp4"
    # Q6videos
    "Q6videos/Q6_Video_3b23e96a-1f6e-4490-bbe2-206b0884d3f8_e58eb9d3-21ac-46ee-b95a-d5a7c3ca21a0.mp4"
    "Q6videos/Q6_Video_68f8e9db-3af0-4d43-9a3d-3cf9e0869d97_742901b1-6f0f-479e-b60c-c72a9f975d90.mp4"
    "Q6videos/Q6_Video_8642b6ca-a3bf-4a74-8f18-ff5aee5589cd_8fbb9a61-71a2-4853-bd69-2514d4c0bb9e.mp4"
    
    # Q8videos
    "Q8videos/Q8_Video_3b23e96a-1f6e-4490-bbe2-206b0884d3f8_e58eb9d3-21ac-46ee-b95a-d5a7c3ca21a0.mp4"
    "Q8videos/Q8_Video_68f8e9db-3af0-4d43-9a3d-3cf9e0869d97_742901b1-6f0f-479e-b60c-c72a9f975d90.mp4"
    "Q8videos/Q8_Video_8642b6ca-a3bf-4a74-8f18-ff5aee5589cd_8fbb9a61-71a2-4853-bd69-2514d4c0bb9e.mp4"
    "Q8videos/Q8_Video_a154eae9-1906-4c2a-b836-ea94708b3eaa_2722e930-992c-4df7-8a63-ef01e7c8b7a0.mp4"
    "Q8videos/Q8_Video_e7021045-c488-44be-a32d-494f59531f93_a759b156-47c9-4503-96d8-349d43eb49e1.mp4"
    "Q8videos/Q8_Video_f3e6c8bb-8e2e-4c33-b95f-e92aa7411189_384e6b12-207e-41a5-97f3-26871c21b02c.mp4"
    
    # Q11videos
    "Q11videos/Q11_video_3b23e96a-1f6e-4490-bbe2-206b0884d3f8_e58eb9d3-21ac-46ee-b95a-d5a7c3ca21a0.mp4"
    "Q11videos/Q11_video_a154eae9-1906-4c2a-b836-ea94708b3eaa_2722e930-992c-4df7-8a63-ef01e7c8b7a0.mp4"
    "Q11videos/Q11_video_f3e6c8bb-8e2e-4c33-b95f-e92aa7411189_384e6b12-207e-41a5-97f3-26871c21b02c.mp4"
)

# Function to process a single video
process_video() {
    local video_path="$1"
    local base_name=$(basename "$video_path" .mp4)
    local dir_name=$(dirname "$video_path")
    
    # Skip if already cleaned
    if [[ "$base_name" == *"_clean" ]]; then
        echo "Skipping already cleaned file: $video_path"
        return
    fi
    
    local input_file="./CONSENTvideos/$video_path"
    local cleaned_file="./CONSENTvideos/$dir_name/${base_name}_clean.mp4"
    local transcript_file="./transcripts/${base_name}.txt"
    
    echo "Processing: $input_file"
    
    # Check if input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "Warning: Input file not found: $input_file"
        return
    fi
    
    # Step 1: Clean the video with ffmpeg
    echo "  -> Cleaning video..."
    if ffmpeg -i "$input_file" -c:v copy -c:a aac -b:a 160k -movflags +faststart "$cleaned_file" -y; then
        echo "  -> Video cleaned successfully"
    else
        echo "  -> Error cleaning video: $input_file"
        return
    fi
    
    # Step 2: Transcribe with insanely-fast-whisper
    echo "  -> Transcribing..."
    if insanely-fast-whisper --file-name "$cleaned_file" --transcript-path "$transcript_file" --device-id mps; then
        echo "  -> Transcription completed: $transcript_file"
    else
        echo "  -> Error transcribing: $cleaned_file"
    fi
    
    echo "  -> Finished processing: $base_name"
    echo ""
}

# Main execution
echo "Starting batch video processing..."
echo "Total videos to process: ${#videos[@]}"
echo ""

# Process each video
for video in "${videos[@]}"; do
    process_video "$video"
done

echo "Batch processing completed!"
echo "Cleaned videos are in their respective folders with '_clean' suffix"
echo "Transcripts are in the ./transcripts/ directory"
