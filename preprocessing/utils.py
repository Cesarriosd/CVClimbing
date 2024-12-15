import cv2
import os
import yt_dlp

def download_youtube_video(youtube_url, output_path):
    """Download a YouTube video as an MP4 file."""
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s')
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        video_path = ydl.prepare_filename(info_dict)
    return video_path

def extract_video_segment(input_path, output_path, start_second, finish_second):
    """Extract a segment of a video and save it to a new file."""
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if start_second >= duration or finish_second > duration:
        raise ValueError("Start or finish time exceeds video duration.")

    # Calculate frame range to extract
    start_frame = int(start_second * fps)
    finish_frame = int(finish_second * fps)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < finish_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    # Input YouTube video URL and parameters
    youtube_url = "https://www.youtube.com/watch?v=aGE4SOOmYlk"
    output_directory = ".//..//res//videos"
    start_minute = 13
    start_second = 42
    finish_minute = 13
    finish_second = 64
    start_time = start_minute*60 + start_second # Define start time in seconds
    finish_time = finish_minute*60 + finish_second    # Define finish time in seconds

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Step 1: Download YouTube video
    #print("Downloading video...")
    downloaded_video_path = download_youtube_video(youtube_url, output_directory)
    #print(f"Video downloaded to {downloaded_video_path}")

    # Step 2: Extract segment
    output_clip_path = os.path.join(output_directory, "example.mp4")
    print("Extracting video segment...")
    extract_video_segment(downloaded_video_path, output_clip_path, start_time, finish_time)
    print(f"Video segment saved to {output_clip_path}")
