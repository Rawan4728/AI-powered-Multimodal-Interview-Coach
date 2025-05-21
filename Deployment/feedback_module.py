import os
import cv2
import subprocess
import numpy as np
import torch
import whisper
from deepface import DeepFace
import mediapipe as mp
import openai

model = whisper.load_model("medium")

def extract_audio_with_ffmpeg(video_path, audio_path="audio.wav"):
    try:
        subprocess.call([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ])
        return audio_path
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return None

def analyze_face_expression(frame):
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        return result[0]["dominant_emotion"]
    except Exception as e:
        print(f"Facial expression analysis error: {e}")
        return "Unknown"

def analyze_body_posture(video_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    posture_issues = 0
    blinking = 0
    total_frames = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                if abs(left.y - right.y) > 0.05:
                    posture_issues += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eye_area = gray[100:150, 250:400]
            if eye_area.mean() < 70:
                blinking += 1

    cap.release()
    posture_score = posture_issues / max(1, total_frames)
    blinking_rate = blinking / max(1, total_frames)
    return posture_score, blinking_rate

# Corrected GPT feedback generation function
def generate_feedback_gpt(transcript, emotion, posture_score, blinking_rate):
    posture_label = "unstable" if posture_score > 0.05 else "stable"
    blinking_label = "frequent" if blinking_rate > 0.08 else "steady"

    prompt = f"""
You're an AI assistant trained to give professional feedback on interview performance.
The candidate showed:
- Facial expression: {emotion}
- Posture: {posture_label}
- Blinking: {blinking_label}
- Transcript: \"{transcript}\"
Please provide:
📉 Behavior Feedback (describe emotional signals, posture meaning, eye contact).
📢 Answer Feedback (evaluate transcript, grammar, clarity, and suggest improvements).
Format your answer using markdown like this:
📉 Behavior Feedback
[Your feedback]
📢 Answer Feedback
[Your feedback]
"""

    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert interview coach."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Main function to process video for transcription and feedback
def process_video(video_file):
    if video_file is None:
        return "❌ No video uploaded", "❌", "❌"

    try:
        video_path = video_file
        audio_path = extract_audio_with_ffmpeg(video_path)
        if not audio_path or not os.path.exists(audio_path):
            return "❌ Failed to extract audio", "❌", "❌"

        transcription = model.transcribe(audio_path)
        text = transcription.get("text", "❌ Failed to transcribe")
        os.remove(audio_path)  # Cleanup audio file after use

        # Capture the first frame from the video to analyze facial expression
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        expression = analyze_face_expression(frame)

        # Analyze body posture and blinking
        posture_score, blinking_rate = analyze_body_posture(video_path)

        # Generate feedback using GPT
        full_feedback = generate_feedback_gpt(text, expression, posture_score, blinking_rate)
        if "📢 Answer Feedback" in full_feedback:
            behavior_feedback, answer_feedback = full_feedback.split("📢 Answer Feedback", 1)
            behavior_feedback = behavior_feedback.strip()
            answer_feedback = "📢 Answer Feedback" + answer_feedback.strip()
        else:
            behavior_feedback = full_feedback
            answer_feedback = "❌ Failed to generate full feedback"

        return text, behavior_feedback, answer_feedback

    except Exception as e:
        print(f"Processing error: {e}")
        return "❌ An error occurred", "❌", "❌"
