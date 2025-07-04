from Utils.extracts_frames import extract_frames_from_video
from Utils.annotated_frames import annotate_frames
import os
import argparse




def main(args):

    video_path = "InputVideos/"  # Replace with your video path
    videos=[f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f)) and f.endswith(('.mp4', '.avi', '.mov'))]
    video_name_with_extension=videos[0]
    videos[0]=videos[0].split('.')[0]
    video_path= os.path.join(video_path, video_name_with_extension)  # Use the first video in the directory
    extractedFrames_path = f"ExtractedFrames/{videos[0]}"  # Replace with your desired output directory
    annotatedFrames_path = f"AnnotatedFrames/{videos[0]}"  # Replace with your desired output directory
    labesl_csv_path = f"Labels/{videos[0]}/labels.csv"  # Path to save the labels CSV file
    step = 50  # Extract every 5th frame
    max_frames = None  # Set to None to extract all frames, or specify a limit

    if args.continue_annotation:
        try:
            print(f"Annotating frames in directory: {extractedFrames_path}")
            annotate_frames(extractedFrames_path, annotatedFrames_path, labesl_csv_path, circle_radius=10, 
                            continue_annotation=True,assisted=args.assisted)
        except Exception as e:
            print(f"An error occurred during annotation: {e}")
        return

    
    try:
        print(f"Extracting frames from video: {videos[0]}, saving to {extractedFrames_path}, step={step}, max_frames={max_frames}")
        saved_paths = extract_frames_from_video(video_path, extractedFrames_path, step, max_frames)
        print(f"Extracted frames saved at: {saved_paths}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    try:
        print(f"Annotating frames in directory: {extractedFrames_path}")
        annotate_frames(extractedFrames_path, annotatedFrames_path, labesl_csv_path, circle_radius=10, 
                        continue_annotation=False, assisted=args.assisted)
    except Exception as e:
        print(f"An error occurred during annotation: {e}")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script para extracción de frames y anotación manual del balón"
    )

    parser.add_argument(
        "--continue_annotation",
        action="store_true",
        help="Sigue con la anotación de frames ya extraídos",
    )
    parser.add_argument(
        "--assisted",
        action="store_true",
        help="Anotacion asistisda por el modelo",
    )

    args = parser.parse_args()
    main(args)