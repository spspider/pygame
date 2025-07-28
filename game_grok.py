import asyncio
import platform
import cv2
import mediapipe as mp
import pygame
import numpy as np
import aubio
import io
import base64
from pygame import mixer

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Pygame
pygame.init()
mixer.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pose Dance Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)

# Load and play music
music_file = "music.mp3"  # Placeholder: replace with your music file path
mixer.music.load(music_file)
mixer.music.play(-1)  # Loop indefinitely

# Aubio beat detection setup
SAMPLE_RATE = 44100
HOP_SIZE = 512
WIN_SIZE = 1024
beat_detector = aubio.tempo("default", WIN_SIZE, HOP_SIZE, SAMPLE_RATE)
audio_file = io.open(music_file, 'rb')
audio_source = aubio.source(music_file, SAMPLE_RATE, HOP_SIZE)
beat_effect_radius = 0
beat_effect_duration = 0

# Infinite road background
road_y = 0
ROAD_SPEED = 5
road_lines = [(WIDTH//2, i*50, 10, 30) for i in range(-10, 12)]  # x, y, width, height

# Second dancer animation frames (simplified keyframe poses)
dance_frames = [
    {
        'left_shoulder': (0.4, 0.3), 'right_shoulder': (0.6, 0.3),
        'left_elbow': (0.3, 0.5), 'right_elbow': (0.7, 0.5),
        'left_wrist': (0.2, 0.7), 'right_wrist': (0.8, 0.7),
        'left_hip': (0.4, 0.6), 'right_hip': (0.6, 0.6),
        'left_knee': (0.4, 0.8), 'right_knee': (0.6, 0.8),
        'left_ankle': (0.4, 1.0), 'right_ankle': (0.6, 1.0)
    },
    {
        'left_shoulder': (0.4, 0.3), 'right_shoulder': (0.6, 0.3),
        'left_elbow': (0.5, 0.4), 'right_elbow': (0.5, 0.4),
        'left_wrist': (0.6, 0.3), 'right_wrist': (0.4, 0.3),
        'left_hip': (0.4, 0.6), 'right_hip': (0.6, 0.6),
        'left_knee': (0.3, 0.7), 'right_knee': (0.7, 0.7),
        'left_ankle': (0.2, 0.9), 'right_ankle': (0.8, 0.9)
    }
]
dance_frame_index = 0
dance_frame_time = 0
DANCE_FRAME_DURATION = 500  # ms per frame

# Open camera
cap = cv2.VideoCapture(0)

def draw_stick_figure(screen, landmarks, offset_x=0, scale=200):
    """Draw a stick figure based on pose landmarks."""
    if not landmarks:
        return
    # Key points to draw
    points = {
        'left_shoulder': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
        'right_shoulder': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        'left_elbow': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
        'right_elbow': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
        'left_wrist': landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
        'right_wrist': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
        'left_hip': landmarks[mp_pose.PoseLandmark.LEFT_HIP],
        'right_hip': landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
        'left_knee': landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
        'right_knee': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
        'left_ankle': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
        'right_ankle': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    }
    # Convert to pixel coordinates
    pixel_points = {k: (int(v.x * scale + offset_x), int(v.y * scale + 100)) for k, v in points.items()}
    # Draw lines
    connections = [
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
    ]
    for start, end in connections:
        if start in pixel_points and end in pixel_points:
            pygame.draw.line(screen, WHITE, pixel_points[start], pixel_points[end], 5)

def draw_animated_dancer(screen, frame, offset_x=400, scale=200):
    """Draw the second dancer from animation frames."""
    pixel_points = {k: (int(v[0] * scale + offset_x), int(v[1] * scale + 100)) for k, v in frame.items()}
    connections = [
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
    ]
    for start, end in connections:
        if start in pixel_points and end in pixel_points:
            pygame.draw.line(screen, YELLOW, pixel_points[start], pixel_points[end], 5)

async def main():
    global road_y, beat_effect_radius, beat_effect_duration, dance_frame_index, dance_frame_time
    clock = pygame.time.Clock()
    FPS = 60

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Не удалось получить кадр с камеры.")
                break

            # Process pose
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            # Pygame rendering
            screen.fill(BLACK)
            # Draw road
            road_y += ROAD_SPEED
            if road_y > 50:
                road_y = 0
                for line in road_lines:
                    line[1] -= 50
            for line in road_lines:
                y = line[1] + road_y
                if 0 <= y <= HEIGHT:
                    pygame.draw.rect(screen, GRAY, (line[0] - line[2]//2, y, line[2], line[3]))

            # Draw player's stick figure
            if results.pose_landmarks:
                draw_stick_figure(screen, results.pose_landmarks.landmark, offset_x=100)

            # Draw animated dancer
            dance_frame_time += clock.get_time()
            if dance_frame_time > DANCE_FRAME_DURATION:
                dance_frame_index = (dance_frame_index + 1) % len(dance_frames)
                dance_frame_time = 0
            draw_animated_dancer(screen, dance_frames[dance_frame_index], offset_x=500)

            # Beat detection
            samples, read = audio_source()
            is_beat = beat_detector(samples)
            if is_beat:
                beat_effect_radius = 50
                beat_effect_duration = 10
            if beat_effect_duration > 0:
                pygame.draw.circle(screen, YELLOW, (WIDTH//2, HEIGHT//2), beat_effect_radius, 2)
                beat_effect_radius += 5
                beat_effect_duration -= 1

            pygame.display.flip()
            clock.tick(FPS)
            await asyncio.sleep(1.0 / FPS)

    cap.release()
    mixer.music.stop()
    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())