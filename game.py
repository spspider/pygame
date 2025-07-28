import cv2
import mediapipe as mp
import pygame
import numpy as np
from aubio import tempo, source

# ---------------------------
# SETTINGS
# ---------------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480
FPS = 30
MUSIC_FILE = 'music.mp3'  # путь к файлу с музыкой

# ---------------------------
# INITIALIZE MEDIA PIPE
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------
# INITIALIZE PYGAME
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((CAM_WIDTH, CAM_HEIGHT))
clock = pygame.time.Clock()

# Загрузка музыки
pygame.mixer.music.load(MUSIC_FILE)
pygame.mixer.music.play()

# ---------------------------
# BEAT DETECTION SETUP (aubio)
# ---------------------------
win_s = 512                 # fft size
hop_s = win_s // 2          # hop size
samplerate = 44100          # default sample rate

beat_o = tempo('default', win_s, hop_s, samplerate)
s = source(MUSIC_FILE, samplerate, hop_s)
tmp_buffer = np.zeros(hop_s, dtype=np.float32)

# ---------------------------
# BACKGROUND (infinite road)
# ---------------------------
bg = pygame.image.load('road.png').convert()
bg = pygame.transform.scale(bg, (CAM_WIDTH, CAM_HEIGHT))
bg_y = 0
speed = 5  # скорость движения дороги

# ---------------------------
# DRAW FUNCTIONS
# ---------------------------
def draw_stickman(surface, landmarks, color=(0, 255, 0)):
    # landmarks: dict of (x, y) points in screen coords
    # рисуем линии между ключевыми точками
    connections = [
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'),
        ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP')
    ]
    for a, b in connections:
        if a in landmarks and b in landmarks:
            pygame.draw.line(surface, color, landmarks[a], landmarks[b], 4)


# Пример анимации второго танцующего персонажа
dance_frames = []  # сюда можно загрузить несколько поз на кадры для цикла
# ... загрузка dance_frames из изображений или процедурно

frame_idx = 0

# ---------------------------
# MAIN LOOP
# ---------------------------
cap = cv2.VideoCapture(0)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    # Подготовка списка точек для отрисовки
    user_landmarks = {}
    if result.pose_landmarks:
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w = CAM_HEIGHT, CAM_WIDTH
            cx, cy = int(lm.x * w), int(lm.y * h)
            user_landmarks[mp_pose.PoseLandmark(id).name] = (cx, cy)

    # ---------------------------
    # BEAT DETECTION
    # ---------------------------
    is_beat = False
    samples, read = s()
    if beat_o(samples):
        is_beat = True

    # ---------------------------
    # DRAW BACKGROUND
    # ---------------------------
    bg_y = (bg_y + speed) % CAM_HEIGHT
    screen.blit(bg, (0, bg_y - CAM_HEIGHT))
    screen.blit(bg, (0, bg_y))

    # ---------------------------
    # DRAW USER STICKMAN
    # ---------------------------
    draw_stickman(screen, user_landmarks, color=(0, 255, 0))

    # ---------------------------
    # DRAW AI DANCER
    # ---------------------------
    # Простейшая смена кадров анимации
    if dance_frames:
        frame = dance_frames[frame_idx]
        screen.blit(frame, (400, 100))
        frame_idx = (frame_idx + 1) % len(dance_frames)

    # ---------------------------
    # BEAT EFFECT
    # ---------------------------
    if is_beat:
        pygame.draw.circle(screen, (255, 255, 0), (CAM_WIDTH//2, CAM_HEIGHT//2), 100, 10)

    # ---------------------------
    # EVENTS
    # ---------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(FPS)

cap.release()
pygame.quit()
