from Utils.extracts_frames import extract_frames_from_video
import os



def main():
    video_path = "InputVideos/"  # Replace with your video path
    videos=[f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f)) and f.endswith(('.mp4', '.avi', '.mov'))]
    video_path= os.path.join(video_path, videos[0])  # Use the first video in the directory
    output_dir = f"ExtractedFrames/{videos[0]}"  # Replace with your desired output directory
    step = 10  # Extract every 5th frame
    max_frames = 50  # Set to None to extract all frames, or specify a limit

    try:
        print(f"Extracting frames from video: {videos[0]}, saving to {output_dir}, step={step}, max_frames={max_frames}")
        saved_paths = extract_frames_from_video(video_path, output_dir, step, max_frames)
        print(f"Extracted frames saved at: {saved_paths}")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()