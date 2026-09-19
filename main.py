import cv2
import numpy as np
import random
import math
import time

from cvzone.HandTrackingModule import HandDetector

# ============================================================
# SETTINGS
# ============================================================
CAMERA_ID = 0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

BOARD_X = 30
BOARD_Y = 150
BOARD_WIDTH = 760
BOARD_HEIGHT = 570

COLS = 4
ROWS = 3

CELL_W = BOARD_WIDTH // COLS
CELL_H = BOARD_HEIGHT // ROWS

TAB = 27
PIECE_W = CELL_W + TAB * 2
PIECE_H = CELL_H + TAB * 2

PANEL_X = 810
PANEL_Y = 215
PANEL_W = 440
PANEL_H = 485

# Keep LIVE and TIMER separate.
LIVE_X = 955
LIVE_Y = 15
LIVE_W = 300
LIVE_H = 185

TIMER_X = 820
TIMER_Y = 55

PINCH_ON = 0.30
PINCH_OFF = 0.48

# Hand cursor smoothing. Higher = faster.
CURSOR_SMOOTH = 0.82

# Piece follows finger almost immediately.
DRAG_SMOOTH = 0.98

SNAP_DISTANCE = 110

# Release only after this many clear non-pinching frames.
RELEASE_FRAMES = 4

# If hand disappears briefly, keep the selected piece.
HAND_LOST_GRACE = 12

TRAY_POSITIONS = []
for r in range(4):
    for c in range(3):
        TRAY_POSITIONS.append(
            (PANEL_X + 75 + c * 135, 285 + r * 105)
        )


# ============================================================
# CAMERA / HAND DETECTOR
# ============================================================
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise SystemExit(
        "Camera could not be opened. Try CAMERA_ID = 1."
    )

detector = HandDetector(
    detectionCon=0.75,
    maxHands=1
)


# ============================================================
# HELPERS
# ============================================================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def make_jigsaw_mask(row, col, vertical, horizontal):
    mask = np.zeros((PIECE_H, PIECE_W), np.uint8)

    cv2.rectangle(
        mask,
        (TAB, TAB),
        (TAB + CELL_W - 1, TAB + CELL_H - 1),
        255,
        -1
    )

    radius = TAB

    # top
    if row > 0:
        value = -horizontal[row][col]
        center = (TAB + CELL_W // 2, TAB)
        cv2.circle(mask, center, radius, 255 if value > 0 else 0, -1)

    # bottom
    if row < ROWS - 1:
        value = horizontal[row + 1][col]
        center = (TAB + CELL_W // 2, TAB + CELL_H)
        cv2.circle(mask, center, radius, 255 if value > 0 else 0, -1)

    # left
    if col > 0:
        value = -vertical[row][col]
        center = (TAB, TAB + CELL_H // 2)
        cv2.circle(mask, center, radius, 255 if value > 0 else 0, -1)

    # right
    if col < COLS - 1:
        value = vertical[row][col + 1]
        center = (TAB + CELL_W, TAB + CELL_H // 2)
        cv2.circle(mask, center, radius, 255 if value > 0 else 0, -1)

    return mask


def crop_piece(photo, row, col):
    padded = cv2.copyMakeBorder(
        photo,
        TAB, TAB, TAB, TAB,
        cv2.BORDER_REPLICATE
    )

    x = col * CELL_W
    y = row * CELL_H

    return padded[y:y + PIECE_H, x:x + PIECE_W].copy()


class Piece:
    def __init__(self, image, mask, target, number):
        self.image = image
        self.mask = mask
        self.target = np.array(target, dtype=np.float32)
        self.position = self.target.copy()
        self.tray_position = self.target.copy()

        self.number = number
        self.locked = False
        self.dragging = False

        # Finger-to-piece offset captured at pinch.
        self.grab_offset = np.zeros(2, dtype=np.float32)

    def contains(self, point, scale=1.0):
        w = int(PIECE_W * scale)
        h = int(PIECE_H * scale)

        left = int(self.position[0] - w / 2)
        top = int(self.position[1] - h / 2)

        lx = int((point[0] - left) / scale)
        ly = int((point[1] - top) / scale)

        if lx < 0 or ly < 0 or lx >= PIECE_W or ly >= PIECE_H:
            return False

        return self.mask[ly, lx] > 0

    def draw(self, screen, scale=1.0, outline=True):
        if scale == 1.0:
            img = self.image
            mask = self.mask
        else:
            w = int(PIECE_W * scale)
            h = int(PIECE_H * scale)

            img = cv2.resize(
                self.image, (w, h),
                interpolation=cv2.INTER_AREA
            )
            mask = cv2.resize(
                self.mask, (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        h, w = mask.shape[:2]

        x = int(self.position[0] - w / 2)
        y = int(self.position[1] - h / 2)

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(screen.shape[1], x + w)
        y1 = min(screen.shape[0], y + h)

        if x0 >= x1 or y0 >= y1:
            return

        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        roi = screen[y0:y1, x0:x1]
        src = img[sy0:sy1, sx0:sx1]
        m = mask[sy0:sy1, sx0:sx1]

        roi[m > 0] = src[m > 0]

        if outline:
            edge = cv2.morphologyEx(
                m,
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), np.uint8)
            )

            if self.dragging:
                color = (0, 255, 255)
            elif self.locked:
                color = (50, 255, 80)
            else:
                color = (225, 225, 225)

            roi[edge > 0] = color


def create_puzzle(photo):
    photo = cv2.resize(
        photo,
        (BOARD_WIDTH, BOARD_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    vertical = [[0] * (COLS + 1) for _ in range(ROWS)]
    horizontal = [[0] * COLS for _ in range(ROWS + 1)]

    for r in range(ROWS):
        for c in range(1, COLS):
            vertical[r][c] = random.choice([-1, 1])

    for r in range(1, ROWS):
        for c in range(COLS):
            horizontal[r][c] = random.choice([-1, 1])

    pieces = []

    for r in range(ROWS):
        for c in range(COLS):
            img = crop_piece(photo, r, c)
            mask = make_jigsaw_mask(
                r, c, vertical, horizontal
            )

            target = (
                BOARD_X + c * CELL_W + CELL_W / 2,
                BOARD_Y + r * CELL_H + CELL_H / 2
            )

            pieces.append(
                Piece(
                    img,
                    mask,
                    target,
                    r * COLS + c + 1
                )
            )

    positions = TRAY_POSITIONS.copy()
    random.shuffle(positions)

    for piece, p in zip(pieces, positions):
        piece.position = np.array(p, np.float32)
        piece.tray_position = piece.position.copy()

    return pieces


def reset_puzzle(pieces):
    positions = TRAY_POSITIONS.copy()
    random.shuffle(positions)

    for piece, p in zip(pieces, positions):
        piece.position = np.array(p, np.float32)
        piece.tray_position = piece.position.copy()
        piece.locked = False
        piece.dragging = False


# ============================================================
# TIMER DIAL
# ============================================================
def draw_timer_dial(screen, selected, pulse):
    cx = WINDOW_WIDTH // 2
    cy = 390
    radius = 190

    cv2.circle(
        screen,
        (cx, cy),
        radius + 12,
        (30, 35, 48),
        -1
    )

    cv2.circle(
        screen,
        (cx, cy),
        radius + 12,
        (80, 180, 255),
        3
    )

    # rotating highlight
    angle = -90 + (selected - 1) * 40
    rad = math.radians(angle)

    hx = int(cx + math.cos(rad) * radius)
    hy = int(cy + math.sin(rad) * radius)

    cv2.circle(
        screen,
        (hx, hy),
        34 + int(4 * pulse),
        (0, 210, 255),
        -1
    )

    for n in range(1, 10):
        angle = -90 + (n - 1) * 40
        rad = math.radians(angle)

        x = int(cx + math.cos(rad) * radius)
        y = int(cy + math.sin(rad) * radius)

        color = (255, 255, 255)

        if n == selected:
            color = (20, 20, 20)

        cv2.putText(
            screen,
            str(n),
            (x - 12, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            color,
            3,
            cv2.LINE_AA
        )

    cv2.circle(
        screen,
        (cx, cy),
        88,
        (20, 25, 35),
        -1
    )

    cv2.putText(
        screen,
        f"{selected}",
        (cx - 22, cy + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        (100, 230, 255),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        screen,
        f"{selected} MINUTE"
        + ("" if selected == 1 else "S"),
        (cx - 92, cy + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (190, 200, 215),
        1,
        cv2.LINE_AA
    )


def draw_setup(screen, step, selected, pulse):
    cv2.putText(
        screen,
        "PUZZLE TIMER",
        (455, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    if step == 0:
        cv2.putText(
            screen,
            "Do you want a timer?",
            (430, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (100, 220, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            screen,
            "Y = YES        N = NO",
            (455, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:
        cv2.putText(
            screen,
            "Choose the time",
            (480, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (100, 220, 255),
            2,
            cv2.LINE_AA
        )

        draw_timer_dial(
            screen,
            selected,
            pulse
        )

        cv2.putText(
            screen,
            "LEFT / RIGHT or UP / DOWN = rotate",
            (375, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (205, 210, 220),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            screen,
            "ENTER = start puzzle",
            (470, 665),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (100, 255, 150),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            screen,
            "ESC = back",
            (540, 710),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (160, 160, 160),
            1,
            cv2.LINE_AA
        )


def draw_confetti(screen, particles):
    for p in particles:
        p["y"] += p["speed"]

        if p["y"] > WINDOW_HEIGHT:
            p["y"] = random.randint(-200, -10)

        cv2.rectangle(
            screen,
            (int(p["x"]), int(p["y"])),
            (
                int(p["x"] + p["size"]),
                int(p["y"] + p["size"] * 2)
            ),
            p["color"],
            -1
        )


# ============================================================
# STATE
# ============================================================
MODE_CAMERA = 0
MODE_SETUP = 1
MODE_PUZZLE = 2

mode = MODE_CAMERA

pieces = []

setup_step = 0
timer_minutes = 1
timer_enabled = False
timer_start = None
timer_remaining = 0

smooth_cursor = None

selected_piece = None

# Robust pinch state
pinch_active = False
non_pinch_frames = 0
hand_lost_frames = 0

completion_start = None

confetti = []


def start_game(photo):
    global pieces
    global mode
    global timer_start
    global timer_remaining
    global selected_piece
    global smooth_cursor
    global pinch_active
    global non_pinch_frames
    global hand_lost_frames
    global completion_start

    pieces = create_puzzle(photo)

    selected_piece = None
    smooth_cursor = None

    pinch_active = False
    non_pinch_frames = 0
    hand_lost_frames = 0

    completion_start = None

    if timer_enabled:
        timer_remaining = timer_minutes * 60
        timer_start = time.perf_counter()
    else:
        timer_remaining = 0
        timer_start = None

    mode = MODE_PUZZLE


# ============================================================
# MAIN LOOP
# ============================================================
while True:

    ok, camera = cap.read()

    if not ok or camera is None:
        continue

    camera = cv2.flip(camera, 1)

    # ========================================================
    # CAMERA SCREEN
    # ========================================================
    if mode == MODE_CAMERA:

        screen = np.zeros(
            (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
            np.uint8
        )

        cv2.putText(
            screen,
            "GESTURE PHOTO PUZZLE",
            (35, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (255, 255, 255),
            3
        )

        preview = cv2.resize(
            camera,
            (720, 540)
        )

        screen[130:670, 35:755] = preview

        cv2.rectangle(
            screen,
            (35, 130),
            (755, 670),
            (255, 170, 30),
            3
        )

        cv2.putText(
            screen,
            "Press C to capture photo",
            (845, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (100, 220, 255),
            2
        )

        cv2.putText(
            screen,
            "Q = Quit",
            (845, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

        cv2.imshow(
            "Gesture Photo Puzzle",
            screen
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("c"), ord("C")):
            mode = MODE_SETUP
            setup_step = 0

        elif key in (ord("q"), ord("Q"), 27):
            break

        continue

    # ========================================================
    # TIMER SETUP
    # ========================================================
    if mode == MODE_SETUP:

        screen = np.zeros(
            (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
            np.uint8
        )

        pulse = (
            math.sin(time.perf_counter() * 5) + 1
        ) / 2

        draw_setup(
            screen,
            setup_step,
            timer_minutes,
            pulse
        )

        cv2.imshow(
            "Gesture Photo Puzzle",
            screen
        )

        key = cv2.waitKey(30) & 0xFF

        if setup_step == 0:

            if key in (ord("y"), ord("Y")):
                timer_enabled = True
                setup_step = 1

            elif key in (ord("n"), ord("N")):
                timer_enabled = False
                start_game(camera)

            elif key == 27:
                mode = MODE_CAMERA

        else:

            # Rotate dial with arrow keys.
            if key in (81, 2424832, ord("a"), ord("A")):
                timer_minutes -= 1

            elif key in (83, 2555904, ord("d"), ord("D")):
                timer_minutes += 1

            elif key in (82, 2490368, ord("w"), ord("W")):
                timer_minutes += 1

            elif key in (84, 2621440, ord("s"), ord("S")):
                timer_minutes -= 1

            timer_minutes = max(
                1,
                min(9, timer_minutes)
            )

            if key in (13, 10):
                start_game(camera)

            elif key == 27:
                setup_step = 0

        continue

    # ========================================================
    # PUZZLE
    # ========================================================

    hands, hand_frame = detector.findHands(
        camera,
        draw=True
    )

    cursor = None
    hand_pinching = False

    if hands:

        hand_lost_frames = 0

        hand = hands[0]
        lm = hand["lmList"]

        index = (lm[8][0], lm[8][1])
        thumb = (lm[4][0], lm[4][1])
        wrist = (lm[0][0], lm[0][1])
        palm = (lm[9][0], lm[9][1])

        palm_size = max(
            dist(wrist, palm),
            1
        )

        pinch_ratio = (
            dist(index, thumb)
            / palm_size
        )

        camera_h, camera_w = camera.shape[:2]

        raw = np.array(
            [
                index[0] * WINDOW_WIDTH / camera_w,
                index[1] * WINDOW_HEIGHT / camera_h
            ],
            dtype=np.float32
        )

        # Prevent the cursor from jumping due to hand noise.
        raw[0] = np.clip(
            raw[0],
            0,
            WINDOW_WIDTH - 1
        )

        raw[1] = np.clip(
            raw[1],
            0,
            WINDOW_HEIGHT - 1
        )

        if smooth_cursor is None:
            smooth_cursor = raw.copy()
        else:
            smooth_cursor += (
                raw - smooth_cursor
            ) * CURSOR_SMOOTH

        cursor = (
            int(smooth_cursor[0]),
            int(smooth_cursor[1])
        )

        # Hysteresis: different thresholds for ON/OFF.
        if pinch_active:
            hand_pinching = (
                pinch_ratio < PINCH_OFF
            )
        else:
            hand_pinching = (
                pinch_ratio < PINCH_ON
            )

    else:

        hand_lost_frames += 1

        # Do NOT immediately release a piece just because
        # MediaPipe misses one frame.
        hand_pinching = pinch_active

        if hand_lost_frames > HAND_LOST_GRACE:
            hand_pinching = False

    # ========================================================
    # START PINCH / SELECT PIECE
    # ========================================================
    if (
        hand_pinching
        and not pinch_active
        and cursor is not None
        and selected_piece is None
    ):

        # Only loose pieces can be selected.
        for piece in reversed(pieces):

            if piece.locked:
                continue

            if piece.contains(cursor, 0.42):

                selected_piece = piece
                piece.dragging = True

                # Capture exact finger-to-piece relationship.
                piece.grab_offset = (
                    piece.position
                    - np.array(
                        cursor,
                        dtype=np.float32
                    )
                )

                break

    # ========================================================
    # DRAG
    # ========================================================
    if (
        selected_piece is not None
        and cursor is not None
        and hand_pinching
    ):

        piece = selected_piece

        desired = (
            np.array(
                cursor,
                dtype=np.float32
            )
            + piece.grab_offset
        )

        # Very fast direct following.
        piece.position += (
            desired - piece.position
        ) * DRAG_SMOOTH

    # ========================================================
    # RELEASE
    # ========================================================
    if selected_piece is not None:

        if hand_pinching:
            non_pinch_frames = 0

        else:
            non_pinch_frames += 1

        # Release only after several clear frames.
        if non_pinch_frames >= RELEASE_FRAMES:

            piece = selected_piece

            piece.dragging = False

            d = dist(
                piece.position,
                piece.target
            )

            if d <= SNAP_DISTANCE:

                piece.position = (
                    piece.target.copy()
                )

                piece.locked = True

            else:

                # Return only AFTER a real release.
                piece.position = (
                    piece.tray_position.copy()
                )

            selected_piece = None
            non_pinch_frames = 0

    # Update state AFTER processing the current frame.
    pinch_active = hand_pinching

    # ========================================================
    # TIMER
    # ========================================================
    if timer_enabled and timer_start is not None:

        elapsed = (
            time.perf_counter()
            - timer_start
        )

        timer_remaining = max(
            0,
            timer_minutes * 60
            - int(elapsed)
        )

    completed = sum(
        1 for p in pieces if p.locked
    )

    complete = (
        len(pieces) > 0
        and completed == len(pieces)
    )

    time_up = (
        timer_enabled
        and timer_remaining <= 0
        and not complete
    )

    if complete and completion_start is None:
        completion_start = time.perf_counter()

        confetti = []
        for _ in range(100):
            confetti.append(
                {
                    "x": random.randint(30, WINDOW_WIDTH - 30),
                    "y": random.randint(-500, 0),
                    "speed": random.uniform(2, 6),
                    "size": random.randint(3, 8),
                    "color": random.choice([
                        (0, 255, 100),
                        (0, 220, 255),
                        (255, 80, 200),
                        (255, 220, 0),
                        (80, 160, 255)
                    ])
                }
            )

    # ========================================================
    # DRAW PUZZLE SCREEN
    # ========================================================
    screen = np.zeros(
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
        np.uint8
    )

    cv2.putText(
        screen,
        "GESTURE PHOTO PUZZLE",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        3
    )

    cv2.putText(
        screen,
        "Pinch = pick | Move = drag | Release = snap",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 210, 255),
        2
    )

    # Board
    cv2.rectangle(
        screen,
        (BOARD_X - 5, BOARD_Y - 5),
        (
            BOARD_X + BOARD_WIDTH + 5,
            BOARD_Y + BOARD_HEIGHT + 5
        ),
        (35, 40, 50),
        -1
    )

    # Grid only while puzzle is unfinished.
    if not complete:

        for r in range(ROWS):
            for c in range(COLS):

                x = BOARD_X + c * CELL_W
                y = BOARD_Y + r * CELL_H

                cv2.rectangle(
                    screen,
                    (x, y),
                    (x + CELL_W, y + CELL_H),
                    (70, 75, 85),
                    1
                )

    # Panel
    cv2.rectangle(
        screen,
        (PANEL_X, PANEL_Y),
        (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
        (25, 30, 40),
        -1
    )

    cv2.rectangle(
        screen,
        (PANEL_X, PANEL_Y),
        (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
        (70, 80, 95),
        2
    )

    cv2.putText(
        screen,
        "PUZZLE PIECES",
        (PANEL_X + 20, PANEL_Y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2
    )

    # Loose pieces first
    for piece in pieces:

        if (
            not piece.locked
            and piece is not selected_piece
        ):

            piece.draw(
                screen,
                0.42,
                True
            )

    # Locked pieces
    for piece in pieces:

        if piece.locked:

            piece.draw(
                screen,
                1.0,
                not complete
            )

    # Selected piece last
    if selected_piece is not None and not complete:

        selected_piece.draw(
            screen,
            1.0,
            True
        )

    # ========================================================
    # TIMER -- SEPARATE FROM LIVE
    # ========================================================
    if timer_enabled and not time_up:

        mm = timer_remaining // 60
        ss = timer_remaining % 60

        timer_text = f"TIME {mm:02d}:{ss:02d}"

        color = (
            (0, 90, 255)
            if timer_remaining <= 30
            else (100, 255, 150)
        )

        cv2.rectangle(
            screen,
            (TIMER_X, TIMER_Y),
            (TIMER_X + 125, TIMER_Y + 55),
            (20, 25, 35),
            -1
        )

        cv2.rectangle(
            screen,
            (TIMER_X, TIMER_Y),
            (TIMER_X + 125, TIMER_Y + 55),
            color,
            2
        )

        cv2.putText(
            screen,
            timer_text,
            (TIMER_X + 10, TIMER_Y + 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # ========================================================
    # LIVE CAMERA
    # ========================================================
    live = cv2.resize(
        hand_frame,
        (LIVE_W, LIVE_H)
    )

    screen[
        LIVE_Y:LIVE_Y + LIVE_H,
        LIVE_X:LIVE_X + LIVE_W
    ] = live

    cv2.rectangle(
        screen,
        (LIVE_X, LIVE_Y),
        (LIVE_X + LIVE_W, LIVE_Y + LIVE_H),
        (255, 150, 30),
        3
    )

    cv2.rectangle(
        screen,
        (LIVE_X + 5, LIVE_Y + 5),
        (LIVE_X + 70, LIVE_Y + 35),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        screen,
        "LIVE",
        (LIVE_X + 15, LIVE_Y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 255),
        2
    )

    # Cursor
    if cursor is not None and not complete and not time_up:

        if pinch_active:

            cv2.circle(
                screen,
                cursor,
                14,
                (0, 255, 255),
                -1
            )

            cv2.circle(
                screen,
                cursor,
                23,
                (255, 255, 255),
                2
            )

        else:

            cv2.circle(
                screen,
                cursor,
                8,
                (0, 255, 255),
                -1
            )

    # Status
    cv2.putText(
        screen,
        f"Completed: {completed}/{len(pieces)}",
        (PANEL_X + 20, PANEL_Y + PANEL_H + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 255, 120),
        2
    )

    # Time up
    if time_up:

        cv2.rectangle(
            screen,
            (280, 300),
            (1000, 480),
            (30, 20, 20),
            -1
        )

        cv2.putText(
            screen,
            "TIME UP!",
            (490, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 80, 255),
            4
        )

    # ========================================================
    # COMPLETION ANIMATION
    # ========================================================
    if complete:

        draw_confetti(
            screen,
            confetti
        )

        elapsed = (
            time.perf_counter()
            - completion_start
        )

        if elapsed < 1.0:

            p = elapsed
            scale = p * p * (3 - 2 * p) * 1.08

        else:

            scale = (
                1.0
                + 0.035 * math.sin(elapsed * 4)
            )

        bw = int(500 * scale)
        bh = int(75 * scale)

        cx = WINDOW_WIDTH // 2
        cy = 105

        x1 = cx - bw // 2
        y1 = cy - bh // 2
        x2 = cx + bw // 2
        y2 = cy + bh // 2

        cv2.rectangle(
            screen,
            (x1, y1),
            (x2, y2),
            (10, 100, 40),
            -1
        )

        cv2.rectangle(
            screen,
            (x1, y1),
            (x2, y2),
            (50, 255, 100),
            3
        )

        text = "PUZZLE COMPLETE!"

        size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * scale,
            2,
        )[0]

        cv2.putText(
            screen,
            text,
            (
                cx - size[0] // 2,
                cy + size[1] // 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * scale,
            (100, 255, 130),
            2,
            cv2.LINE_AA
        )

    # Controls
    cv2.putText(
        screen,
        "C = New Photo",
        (830, 765),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1
    )

    cv2.putText(
        screen,
        "R = Reset",
        (990, 765),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1
    )

    cv2.putText(
        screen,
        "Q = Quit",
        (1100, 765),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1
    )

    cv2.imshow(
        "Gesture Photo Puzzle",
        screen
    )

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q"), 27):
        break

    if key in (ord("c"), ord("C")):
        mode = MODE_SETUP
        setup_step = 0

    elif key in (ord("r"), ord("R")):
        reset_puzzle(pieces)
        selected_piece = None
        pinch_active = False
        non_pinch_frames = 0
        smooth_cursor = None

        if timer_enabled:
            timer_start = time.perf_counter()
            timer_remaining = timer_minutes * 60

        completion_start = None


cap.release()
cv2.destroyAllWindows()