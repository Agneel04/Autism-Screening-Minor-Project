import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
IMG_SIZE        = 224
MODEL_PATH      = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\model.h5"
CAPTURE_SECONDS = 5
FRAMES_TARGET   = 50
OUTPUT_PATH     = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\screening_result.png"

# Severity bands
SEVERITY_BANDS = {
    "High"        : (0.85, 1.00),
    "Medium"      : (0.65, 0.84),
    "Low"         : (0.50, 0.64),
    "Non-Autistic": (0.00, 0.49),
}

SEVERITY_COLORS = {
    "High"        : "#E57373",
    "Medium"      : "#FFB74D",
    "Low"         : "#FFF176",
    "Non-Autistic": "#81C784",
}

# Facial feature thresholds per severity
FEATURE_THRESHOLDS = {
    "blink_rate": {
        "label"       : "Blink Rate (per min)",
        "normal"      : "10–25",
        "High"        : lambda v: v < 8   or v > 28,
        "Medium"      : lambda v: v < 10  or v > 25,
        "Low"         : lambda v: v < 12  or v > 23,
        "Non-Autistic": lambda v: 10 <= v <= 25,
    },
    "eye_detected_ratio": {
        "label"       : "Eye Detection Ratio",
        "normal"      : "> 0.70",
        "High"        : lambda v: v < 0.40,
        "Medium"      : lambda v: v < 0.55,
        "Low"         : lambda v: v < 0.70,
        "Non-Autistic": lambda v: v >= 0.70,
    },
    "symmetry": {
        "label"       : "Facial Asymmetry (%)",
        "normal"      : "< 5%",
        "High"        : lambda v: v > 0.15,
        "Medium"      : lambda v: v > 0.10,
        "Low"         : lambda v: v > 0.05,
        "Non-Autistic": lambda v: v <= 0.05,
    },
    "expr_var": {
        "label"       : "Expression Variability",
        "normal"      : "> 50.0",
        "High"        : lambda v: v < 20.0,
        "Medium"      : lambda v: v < 35.0,
        "Low"         : lambda v: v < 50.0,
        "Non-Autistic": lambda v: v >= 50.0,
    },
    "head_mov": {
        "label"       : "Head Movement (px)",
        "normal"      : "3–30 px",
        "High"        : lambda v: v > 45  or v < 2,
        "Medium"      : lambda v: v > 35  or v < 2.5,
        "Low"         : lambda v: v > 30  or v < 3,
        "Non-Autistic": lambda v: 3 <= v <= 30,
    },
}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL AND OPENCV CASCADES
# ─────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)
print("Model loaded successfully\n")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def cnn_predict(frame_rgb):
    img  = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    prob = model.predict(
        img.reshape(1, IMG_SIZE, IMG_SIZE, 3), verbose=0
    )[0][0]
    return float(prob)

def classify_severity(prob):
    if   prob >= 0.85: return "High"
    elif prob >= 0.65: return "Medium"
    elif prob >= 0.50: return "Low"
    else:              return "Non-Autistic"

def detect_face_and_eyes(gray_frame):
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None, False, 0.0

    x, y, w, h   = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi_gray = gray_frame[y:y+h, x:x+w]

    eyes          = eye_cascade.detectMultiScale(
        face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    eyes_detected = len(eyes) >= 2

    left_half  = face_roi_gray[:, :w//2]
    right_half = face_roi_gray[:, w//2:]
    right_flip = cv2.flip(right_half, 1)
    min_w      = min(left_half.shape[1], right_flip.shape[1])
    diff       = np.abs(
        left_half[:, :min_w].astype(float) -
        right_flip[:, :min_w].astype(float)
    )
    symmetry_score = float(np.mean(diff) / 255.0)
    face_center    = (x + w // 2, y + h // 2)
    return face_center, eyes_detected, symmetry_score

def compute_expression_variance(face_regions):
    if len(face_regions) < 2:
        return 0.0
    resized = [cv2.resize(f, (64, 64)).astype(float) for f in face_regions]
    diffs   = [np.mean(np.abs(resized[i] - resized[i-1]))
               for i in range(1, len(resized))]
    return float(np.mean(diffs))

def generate_feature_report(metrics, severity):
    report = []
    for key, thresholds in FEATURE_THRESHOLDS.items():
        val     = metrics[key]
        flag_fn = thresholds.get(severity)
        flagged = flag_fn(val) if flag_fn else False

        if key == "blink_rate":
            display_val = f"{val:.1f}/min"
        elif key == "eye_detected_ratio":
            display_val = f"{val:.2f}"
        elif key == "symmetry":
            display_val = f"{val * 100:.1f}%"
        elif key == "expr_var":
            display_val = f"{val:.2f}"
        elif key == "head_mov":
            display_val = f"{val:.1f}px"
        else:
            display_val = f"{val:.2f}"

        report.append({
            "label"   : thresholds["label"],
            "value"   : display_val,
            "normal"  : thresholds["normal"],
            "abnormal": flagged,
            "raw"     : val,
        })
    return report

# ─────────────────────────────────────────────────────────────
# RESULT DISPLAY
# ─────────────────────────────────────────────────────────────
def display_results(frames_rgb, cnn_probs, avg_prob, severity, feature_report):
    color = SEVERITY_COLORS[severity]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.6, wspace=0.4)

    # ── Row 1: 5 evenly sampled frames from all 50 ──
    sample_indices = np.linspace(0, len(frames_rgb) - 1, 5, dtype=int)
    for plot_i, frame_i in enumerate(sample_indices):
        ax    = fig.add_subplot(gs[0, plot_i])
        ax.imshow(frames_rgb[frame_i])
        sev_i = classify_severity(cnn_probs[frame_i])
        col_i = SEVERITY_COLORS[sev_i]
        ax.set_title(
            f"Frame {frame_i+1}\np={cnn_probs[frame_i]:.2f}\n{sev_i}",
            color=col_i, fontsize=8, fontweight="bold"
        )
        ax.axis("off")

    # ── Row 2 left: severity gauge ──
    ax_gauge = fig.add_subplot(gs[1, :3])
    ax_gauge.set_facecolor("#16213e")
    bands = [
        ("Non-Autistic", 0.00, 0.50, "#81C784"),
        ("Low",          0.50, 0.65, "#FFF176"),
        ("Medium",       0.65, 0.85, "#FFB74D"),
        ("High",         0.85, 1.00, "#E57373"),
    ]
    for name, lo, hi, c in bands:
        ax_gauge.barh(0, hi - lo, left=lo, height=0.5, color=c, alpha=0.9)
        ax_gauge.text(
            (lo + hi) / 2, 0, name,
            ha="center", va="center",
            fontsize=9, color="#1a1a2e", fontweight="bold"
        )
    ax_gauge.axvline(avg_prob, color="white", linewidth=3, linestyle="--")
    ax_gauge.text(
        avg_prob, 0.32, f"  {avg_prob:.2f}",
        color="white", fontsize=11, fontweight="bold"
    )
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(-0.5, 0.7)
    ax_gauge.axis("off")

    result_text = (
        "Non-Autistic"
        if severity == "Non-Autistic"
        else f"Autistic — {severity} Severity"
    )
    ax_gauge.set_title(
        f"Result: {result_text}   |   Avg Confidence: {avg_prob*100:.1f}%  |  Frames: {len(frames_rgb)}",
        color=color, fontsize=13, fontweight="bold", pad=10
    )

    # ── Row 2 right: probability trend line across all 50 frames ──
    ax_bar = fig.add_subplot(gs[1, 3:])
    ax_bar.set_facecolor("#16213e")
    ax_bar.plot(
        range(1, len(cnn_probs) + 1),
        [p * 100 for p in cnn_probs],
        color="#64B5F6", linewidth=1.2, alpha=0.8, label="Per-frame prob"
    )
    ax_bar.axhline(
        avg_prob * 100, color="white", linestyle="--",
        linewidth=1.5, label=f"Avg: {avg_prob*100:.1f}%"
    )
    ax_bar.axhline(50, color="#aaaaaa", linestyle=":", linewidth=1)
    ax_bar.fill_between(
        range(1, len(cnn_probs) + 1),
        [p * 100 for p in cnn_probs],
        50, alpha=0.15, color="#64B5F6"
    )
    ax_bar.set_xlabel("Frame", color="white")
    ax_bar.set_ylabel("Autism Probability (%)", color="white")
    ax_bar.set_ylim(0, 110)
    ax_bar.tick_params(colors="white")
    ax_bar.set_title(f"Probability Trend — All {len(cnn_probs)} Frames",
                     color="white", fontsize=10)
    ax_bar.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444")

    # ── Row 3: facial feature table ──
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.set_facecolor("#16213e")
    ax_table.axis("off")

    col_labels  = ["Facial Feature", "Measured Value", "Normal Range", "Status"]
    table_data  = []
    cell_colors = []

    for fr in feature_report:
        status = "  Abnormal" if fr["abnormal"] else "  Normal"
        table_data.append([fr["label"], fr["value"], fr["normal"], status])
        row_color = ["#2a2a4a", "#2a2a4a", "#2a2a4a"]
        row_color.append("#4a1a1a" if fr["abnormal"] else "#1a3a1a")
        cell_colors.append(row_color)

    tbl = ax_table.table(
        cellText    = table_data,
        colLabels   = col_labels,
        cellLoc     = "center",
        loc         = "center",
        cellColours = cell_colors
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#555555")
        if row == 0:
            cell.set_facecolor("#0f3460")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_text_props(color="white")

    ax_table.set_title(
        "Facial Feature Analysis", color="white", fontsize=11, pad=12
    )

    plt.suptitle(
        "Autism Screening System — Result Report",
        color="white", fontsize=15, fontweight="bold", y=0.99
    )

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nResult saved as screening_result.png")

# ─────────────────────────────────────────────────────────────
# MAIN SCREENING FUNCTION
# ─────────────────────────────────────────────────────────────
def run_screening():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("=" * 55)
    print("  AUTISM SCREENING SYSTEM")
    print("=" * 55)
    print(f"  Webcam active — look directly at the camera.")
    print(f"  Capturing {FRAMES_TARGET} frames over {CAPTURE_SECONDS} seconds.")
    print(f"  Press Q to quit early.\n")

    frames_rgb        = []
    face_regions_gray = []
    blink_count       = 0
    prev_eyes         = True
    face_centers      = []
    sym_values        = []
    eye_detect_frames = 0
    total_frames      = 0

    capture_interval = CAPTURE_SECONDS / FRAMES_TARGET   # 0.1 seconds
    start_time       = time.time()
    next_capture     = start_time

    while time.time() - start_time < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display   = frame.copy()
        total_frames += 1

        face_center, eyes_detected, sym_score = detect_face_and_eyes(gray)

        if face_center is not None:
            face_centers.append(face_center)
            sym_values.append(sym_score)

            if eyes_detected:
                eye_detect_frames += 1
                if not prev_eyes:
                    blink_count += 1
                prev_eyes = True
            else:
                prev_eyes = False

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) > 0:
                x, y, w, h = sorted(
                    faces, key=lambda f: f[2]*f[3], reverse=True
                )[0]
                face_regions_gray.append(gray[y:y+h, x:x+w])
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            status_text = "Eyes OK" if eyes_detected else "Eyes closed/away"
            cv2.putText(display, f"Blinks: {blink_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display, f"Status: {status_text}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display, f"Symmetry: {sym_score*100:.1f}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected — move closer",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        elapsed   = time.time() - start_time
        remaining = CAPTURE_SECONDS - int(elapsed)
        cv2.putText(
            display,
            f"Time: {remaining}s  |  Frames: {len(frames_rgb)}/{FRAMES_TARGET}",
            (frame.shape[1] - 310, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
        )
        cv2.imshow("Autism Screening — Look at the camera", display)

        # Capture at 10 frames/sec to reach 50 in 5 seconds
        if time.time() >= next_capture:
            frames_rgb.append(frame_rgb.copy())
            if len(frames_rgb) % 10 == 0:
                print(f"  {len(frames_rgb)} frames captured at t={elapsed:.1f}s")
            next_capture += capture_interval

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Capture stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not frames_rgb:
        print("No frames captured. Please check your webcam.")
        return

    # ── CNN predictions on all 50 frames ──
    print(f"\nAnalysing {len(frames_rgb)} frames with CNN model...")
    print("  This may take 20–30 seconds on CPU, please wait...\n")
    cnn_probs = [cnn_predict(f) for f in frames_rgb]
    avg_prob  = float(np.mean(cnn_probs))
    severity  = classify_severity(avg_prob)

    # ── Aggregate facial metrics ──
    blink_rate         = blink_count * (60.0 / CAPTURE_SECONDS)
    eye_detected_ratio = eye_detect_frames / total_frames if total_frames > 0 else 0.0
    expr_var           = compute_expression_variance(face_regions_gray)

    if len(face_centers) > 1:
        pos_arr  = np.array(face_centers)
        head_mov = float(np.max(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)))
    else:
        head_mov = 0.0

    metrics = {
        "blink_rate"        : blink_rate,
        "eye_detected_ratio": eye_detected_ratio,
        "symmetry"          : float(np.mean(sym_values)) if sym_values else 0.0,
        "expr_var"          : expr_var,
        "head_mov"          : head_mov,
    }

    feature_report = generate_feature_report(metrics, severity)

    # ── Console report ──
    print("\n" + "=" * 55)
    print("         AUTISM SCREENING REPORT")
    print("=" * 55)
    result_label = (
        "Non-Autistic"
        if severity == "Non-Autistic"
        else f"Autistic — {severity} Severity"
    )
    print(f"  Result          : {result_label}")
    print(f"  CNN Probability : {avg_prob:.4f}  ({avg_prob*100:.1f}%)")
    print(f"  Severity Band   : {SEVERITY_BANDS[severity]}")
    print(f"  Frames analysed : {len(frames_rgb)}")
    print(f"  Prob Std Dev    : {np.std(cnn_probs):.4f}  (lower = more consistent)")
    print(f"  Prob Range      : {min(cnn_probs):.4f} – {max(cnn_probs):.4f}")
    print()
    print("  Facial Feature Summary:")
    for fr in feature_report:
        status = "ABNORMAL" if fr["abnormal"] else "Normal  "
        print(f"    [{status}]  {fr['label']:30s}: {fr['value']:10s}  (normal: {fr['normal']})")
    print()
    print("  IMPORTANT: This tool is a screening aid only.")
    print("  It is NOT a clinical diagnosis. Please consult")
    print("  a qualified medical professional for assessment.")
    print("=" * 55)

    display_results(frames_rgb, cnn_probs, avg_prob, severity, feature_report)


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_screening()

