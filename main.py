import whisper
import subprocess
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

import cv2
import dlib
import speech_recognition as sr
from moviepy.editor import ImageSequenceClip


source_dir = "originalVideo" 
dest_dir = "finishedVideo"

def main():
    for file in os.listdir(source_dir):
        video_filename = file 
        print(f"Uploaded file: {video_filename}")

        audio_filename = "extracted_audio.wav"
        print("Extracting audio...")
        subprocess.run([
            "ffmpeg", "-i", video_filename, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", audio_filename, "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Finished extracting: {audio_filename}")

        print("Loading Whisper model...")
        model = whisper.load_model("base") 
        print("Finished Loading.")

        print("Start voice recognition...")
        result = model.transcribe(audio_filename, language='ko')
        print("Finished recognition.")

        import re

        transcript = result["text"]
        segments = result["segments"]

        # Patterns indicating sentence end (. ? !)
        sentence_end_pattern = re.compile(r'[.!?]')

        sentences = []
        current_sentence = ""
        sentence_start = None
        for segment in segments:
            text = segment["text"].strip()
            if not text:
                continue
            if sentence_start is None:
                sentence_start = segment["start"]
            current_sentence += " " + text
            if sentence_end_pattern.search(text):
                # Sentence is end
                sentence_end = segment["end"]
                sentences.append({
                    "text": current_sentence.strip(),
                    "start": sentence_start,
                    "end": sentence_end
                })
                current_sentence = ""
                sentence_start = None

        # If last sentence does not contain end pattern. Add last sentence
        if current_sentence:
            sentences.append({
                "text": current_sentence.strip(),
                "start": sentence_start,
                "end": segments[-1]["end"]
            })

        print(f"The number of sentences are {len(sentences)}.")

        # Check Segmented sentences.
        # for i in sentences:
        #   print(i)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        modulated_time = []

        video = VideoFileClip(video_filename)
        fps = video.fps
        print("Video Segmenting...")
        for idx, sentence in enumerate(tqdm(sentences, desc="Segmenting")):
            start_time = sentence["start"] + 0.75
            end_time = sentence["end"] + 0.5   # Add extra time
            # Modulate not to exceed original video duration
            end_time = min(end_time, video.duration)
            # If there is no voice between sentences, skip
            if end_time - start_time < 0.5:  # Ignore less than 0.5 second
                continue
            output_path = os.path.join(output_dir, f"sentence_{idx+1}.mp4")
            ffmpeg_extract_subclip(video_filename, start_time, end_time, targetname=output_path)
            print(f"start time: {start_time} // end time: {end_time}")
            modulated_time.append({
                    "start": start_time,
                    "end": end_time
                })

        print(f"Finished segmentation. Saved in folder '{output_dir}'.")

        # Setting face detection model (HOG)
        detector = dlib.get_frontal_face_detector()


        entire_video = []
        video_order = []
        target_size = (300, 300)
        first_frame = True

        padding = 20 # Padding of 20 pixels around the face

        # Exctract Face in Video
        for root, dirs, files_in_dir in os.walk(output_dir):
            for file in files_in_dir:
                video_path = output_dir + '/' + file
                video_order.append(video_path)
                cap = cv2.VideoCapture(video_path)
                frames = []
                x, y, w, h = (0, 0, 0, 0)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    original_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detections = detector(original_frame)

                    for detection in detections: # Face is in the video
                        if first_frame == True:
                            x, y, w, h = (detection.left(), detection.top(), detection.width(), detection.height())
                            x = max(0, x-padding)
                            y = max(0, y-padding)
                            w = min(frame.shape[1]-x, w+2*padding)
                            h = min(frame.shape[0]-y, h+2*padding)
                            first_frame = False
                        face_img = original_frame[y:y+h, x:x+w]
                        if face_img.size > 0: # check whether face img is empty or not
                            face_img = cv2.resize(face_img, target_size)
                            frames.append(face_img)
                    entire_video.append(frames)
                    print(f"Face Detected: {len(entire_video)}. size = [{x, y, w, h}]")
                    cap.release()
                    first_frame = True

        final_dir = "finishedVideo"
        os.makedirs(final_dir, exist_ok=True)

        #(modulated_time[i]["start"], modulated_time[i]["end"]

        for i, frs in enumerate(entire_video):
            current_original_video_path = video_order[i]
            current_video = VideoFileClip(current_original_video_path)
            clip = ImageSequenceClip(frs, fps=fps)  # FPS
            audio_clip = current_video.subclip().audio
            video_with_audio = clip.set_audio(audio_clip)
            video_path = os.path.join(final_dir, f"video_{i+1}.mp4")
            video_with_audio.write_videofile(video_path, codec='libx264')




