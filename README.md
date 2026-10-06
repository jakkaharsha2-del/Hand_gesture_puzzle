🧩 Gesture Photo Puzzle

Turn any photo into an interactive jigsaw puzzle — controlled entirely with your hand gestures.

A computer-vision-based puzzle game built with Python, OpenCV, MediaPipe hand tracking via cvzone, NumPy, and Pyttsx3.

Capture a photo using your webcam, choose whether you want a timer, and solve the generated 4 × 3 jigsaw puzzle by pinching and dragging pieces with your index finger and thumb.

✨ Features

📸 Capture Your Own Photo

Press C to capture the current webcam frame.

The exact captured image is used to generate the puzzle.

No unexpected camera recapture occurs when starting or resetting.

🧩 Automatic Jigsaw Generation

Converts the captured image into a 12-piece jigsaw puzzle.

Randomized puzzle-piece positions.

Random tabs and slots create an authentic jigsaw appearance.

🖐️ Gesture-Based Controls

Use your index finger as a cursor.

Pinch your thumb and index finger to grab a piece.

Move your hand to drag the piece.

Release the pinch to place it.

🎯 Smart Piece Placement

Pieces automatically snap into position when released close enough to their target.

Incorrect placements return to their tray position.

⏱️ Optional Timer

Choose between 1–9 minutes.

Timer can be enabled or disabled.

Voice announcements notify you when time is running out.

🔊 Sound Effects

Pleasant sound for a correct placement.

Different sound for an incorrect placement.

Uses Windows winsound.

🗣️ Voice Feedback

Announces remaining time.

Announces "Time Up!"

Celebrates completion with "Hurray! Puzzle complete!"

🎉 Completion Animation

Confetti animation.

Colorful particle effects.

Animated "PUZZLE COMPLETE!" message.

Timer stops immediately after completion.

🔄 Quick Reset

Press R to restart the current puzzle.

The same captured photograph is reused.

No new photo is taken.

📷 Live Camera Preview

A live webcam feed remains visible while playing.

Hand tracking is displayed in the camera preview.

🎮 How It Works

The game follows a simple flow:

       ┌──────────────────┐
       │   Start Program  │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │   Camera Screen  │
       └────────┬─────────┘
                │
             Press C
                │
                ▼
       ┌──────────────────┐
       │ Capture Photo 📸 │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │  Timer Setup ⏱️  │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Generate Puzzle  │
       │     4 × 3        │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Gesture Gameplay │
       └────────┬─────────┘
                │
         Pinch & Drag 🖐️
                │
                ▼
       ┌──────────────────┐
       │ Place All Pieces │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ 🎉 Completion 🎉 │
       └──────────────────┘

🛠️ Technologies Used
Technology	Purpose
🐍 Python	Main programming language
👁️ OpenCV	Camera input, image processing and rendering
🖐️ cvzone	Hand tracking interface
🔢 NumPy	Image and numerical operations
🔊 winsound	Correct/incorrect placement sounds
🗣️ pyttsx3	Offline text-to-speech
🧵 threading	Prevents speech and sounds from freezing the UI
🎲 random	Puzzle randomization and effects
⏱️ time	Timer and animation timing
📐 math	Distance, angles and animation calculations
📋 Requirements
Software

Recommended:

Python 3.9+

Windows OS

Webcam

Note: The project currently uses winsound, which is Windows-specific.

📦 Installation
1. Clone the repository
git clone https://github.com/your-username/gesture-photo-puzzle.git
cd gesture-photo-puzzle

2. Create a virtual environment
python -m venv venv

3. Activate the environment
Windows CMD
venv\Scripts\activate

Windows PowerShell
venv\Scripts\Activate.ps1

4. Install dependencies
pip install opencv-python numpy cvzone mediapipe pyttsx3

▶️ Running the Project

Run the Python file:

python main.py


Make sure your webcam is connected and accessible.

🎮 Controls
Camera Screen
Key	Action
C	Capture a photo
Q	Quit
Timer Setup
Key	Action
Y	Enable timer
N	Disable timer
← / →	Change timer
↑ / ↓	Change timer
ENTER	Start puzzle
ESC	Go back

The timer can be set from:

1 minute → 9 minutes

Puzzle Screen
Key	Action
C	Capture a new photo
R	Restart using the same photo
Q	Quit
Hand Gestures
Gesture	Action
☝️ Index finger	Move cursor
🤏 Pinch	Grab a puzzle piece
✋ Move while pinching	Drag piece
🖐️ Release pinch	Drop piece
🧩 Puzzle Mechanics

The captured image is resized to:

760 × 570 pixels


The board is divided into:

4 columns × 3 rows


Result:

4 × 3 = 12 puzzle pieces


Each piece receives:

A unique jigsaw mask

A target position

A randomized tray position

A unique piece number

🎯 Placement System

When a piece is released, the program calculates the distance between its current position and its target:

d = dist(
    piece.position,
    piece.target
)


If:

distance ≤ 110 pixels


the piece is considered correctly placed.

The piece then:

Snaps into its target.

Becomes locked.

Produces green particle effects.

Plays the correct-placement sound.

If the piece is too far away:

Red particles appear.

An incorrect sound plays.

The piece returns to its tray position.

🖐️ Hand Tracking

The project uses HandDetector from cvzone:

detector = HandDetector(
    detectionCon=0.75,
    maxHands=1
)


The program tracks:

Index fingertip — landmark 8

Thumb tip — landmark 4

Wrist — landmark 0

Palm — landmark 9

The distance between the thumb and index finger is used to determine whether the user is pinching.

🎚️ Pinch Detection

The program uses two thresholds to make the gesture more stable:

PINCH_ON = 0.30
PINCH_OFF = 0.48


This creates pinch hysteresis, reducing accidental grabbing and releasing caused by small hand movements.

A piece is released only after:

RELEASE_FRAMES = 4


clear non-pinching frames.

🎯 Cursor Smoothing

Hand movement can naturally contain small amounts of noise.

The project therefore smooths the cursor:

CURSOR_SMOOTH = 0.82


Piece movement uses:

DRAG_SMOOTH = 0.98


This provides fast but relatively stable piece movement.

⏱️ Timer System

The timer is optional.

When enabled, the selected duration is converted into seconds:

timer_remaining = timer_minutes * 60


The program continuously calculates elapsed time using:

time.perf_counter()


The timer also provides voice announcements.

Example:

"5 minutes left."
"4 minutes left."
"3 minutes left."
"2 minutes left."
"1 minute left."


When the timer reaches zero:

TIME UP!


is displayed and:

"Time Up!"


is spoken.

🗣️ Voice Feedback

The project uses pyttsx3 for offline text-to-speech.

Speech runs inside a background thread so that the OpenCV interface remains responsive.

Example:

speak(
    "Hurray! Puzzle complete!"
)


A lock prevents multiple speech engines from running at the same time.

🔊 Sound Effects

Correct placement:

880 Hz → 1175 Hz


Incorrect placement:

300 Hz → 220 Hz


Sounds are executed in background threads so they don't interrupt gameplay.

🎉 Completion System

The puzzle is considered complete when:

completed == len(pieces)


Once all pieces are locked:

⏱️ Timer immediately stops.

🗣️ Completion announcement plays.

🎊 Confetti starts.

🟢 Completion banner appears.

✨ Animation begins.

🧩 All puzzle pieces remain locked.

The timer cannot continue counting after completion.

🔄 Reset Behavior

Pressing:

R


does not take another photograph.

Instead, the program calls:

restart_current_puzzle()


which reuses:

captured_photo.copy()


This means you can replay the same photo without returning to the camera screen.

📸 Photo Capture Behavior

When C is pressed:

captured_photo = camera.copy()


The exact current camera frame is stored.

That image is then used for:

Puzzle generation

Resetting the puzzle

Replaying the current puzzle

This prevents the puzzle image from unexpectedly changing.

⚙️ Configuration

Most gameplay settings can be changed near the top of the program.

Camera
CAMERA_ID = 0


Try:

CAMERA_ID = 1


if the default camera cannot be opened.

Puzzle Size
COLS = 4
ROWS = 3


For example, changing these values can create a different number of pieces.

Snap Distance
SNAP_DISTANCE = 110


Increase this value to make placement easier.

Decrease it to make placement more precise.

Hand Tracking
PINCH_ON = 0.30
PINCH_OFF = 0.48


These values control pinch sensitivity.

🗂️ Suggested Project Structure
gesture-photo-puzzle/
│
├── main.py
├── README.md
├── requirements.txt
│
└── assets/
    └── screenshots/

📄 requirements.txt

You can create a requirements.txt file containing:

opencv-python
numpy
cvzone
mediapipe
pyttsx3


Then install everything with:

pip install -r requirements.txt


winsound does not need to be installed separately because it is included with standard Windows Python installations.

🐛 Troubleshooting
Camera does not open

Try changing:

CAMERA_ID = 0


to:

CAMERA_ID = 1


Also make sure another application isn't currently using the webcam.

Hand tracking is inaccurate

Try adjusting:

detectionCon=0.75


You can also improve lighting and keep your hand clearly visible against the background.

Puzzle pieces are difficult to grab

Try increasing:

PINCH_ON


slightly or increasing the piece detection scale.

Pieces snap too easily

Reduce:

SNAP_DISTANCE = 110


For example:

SNAP_DISTANCE = 80

Voice does not work

Make sure:

Your computer has an available audio output.

pyttsx3 is installed.

Windows speech components are available.

Try:

pip install --upgrade pyttsx3

Sound effects don't work

The project uses:

import winsound


Therefore, the sound-effect system is designed specifically for Windows.

🚀 Future Improvements

Possible upgrades include:

🧩 Adjustable puzzle sizes such as 3×3, 4×4 and 5×5

🏆 Score and leaderboard system

⭐ Difficulty levels

💾 Save puzzle progress

🎵 Background music

📊 Completion statistics

🖼️ Image selection from files

🌐 Cross-platform audio support

🖐️ Multi-hand controls

🎨 Custom puzzle themes

⏱️ Best-time tracking

🥇 High-score system

📱 Touchscreen support

🎥 Demo Flow

A typical game session looks like:

📷 Open camera
      ↓
Press C
      ↓
📸 Capture photo
      ↓
⏱️ Choose timer
      ↓
🧩 Generate 12 pieces
      ↓
🖐️ Pinch a piece
      ↓
↔️ Drag it
      ↓
🖐️ Release
      ↓
✅ Correct → Lock
❌ Incorrect → Return to tray
      ↓
Repeat
      ↓
🎉 PUZZLE COMPLETE!

🔐 Important Notes

The application processes the camera feed locally.

No internet connection is required for normal gameplay.

The captured photo is stored in memory while the application is running.

The project does not automatically save captured photographs to disk.

Only one hand is tracked at a time.

The current sound implementation is Windows-specific.

👨‍💻 Author

Gesture Photo Puzzle

A computer-vision project combining:

Computer Vision + Hand Tracking + Gesture Interaction + Puzzle Gaming + Voice Feedback

⭐ Project Highlights
┌───────────────────────────────────────────┐
│           🧩 GESTURE PHOTO PUZZLE         │
├───────────────────────────────────────────┤
│                                           │
│   📸 Capture       🖐️ Control             │
│   🧩 12 Pieces     ⏱️ Timer               │
│   🎯 Snap          🔊 Sound               │
│   🗣️ Voice         🎉 Confetti            │
│                                           │
│        PLAY WITH YOUR HANDS!              │
│                                           │
└───────────────────────────────────────────┘

📜 License

Add your preferred license here, for example:

MIT License


If this is a personal or academic project, you can also replace this section with your institution/project-specific licensing information.
