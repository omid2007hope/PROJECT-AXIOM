import subprocess
import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

scenarios = [
    {
        "name": "1. Main Menu Exit [5]",
        "inputs": "5\n",
        "expected": ["Welcome to AXIOM", "> "]
    },
    {
        "name": "2. Main Menu Tests [3]",
        "inputs": "3\n\n",
        "expected": ["Welcome to AXIOM", "--- test_path AI ---", "test_embeddings AI"]
    },
    {
        "name": "3. Main Menu Benchmark [4]",
        "inputs": "4\n\n",
        "expected": [
            "AXIOM HARDWARE BENCHMARK & KERNEL SATURATION",
            "INFERENCE DISPATCH LATENCY",
            "TRANSFORMER KERNEL SATURATION"
        ]
    },
    {
        "name": "4. Main Menu Inference [1]",
        "inputs": "1\nHello Axiom\nexit\n",
        "expected": [
            "AXIOM INFERENCE ENGINE",
            "Model Architecture: Single Dedicated Multimodal Perceptron",
            "[Tokens]",
            "[Prediction]",
            "Exiting inference engine."
        ]
    },
    {
        "name": "5. Training: Text (No Media)",
        "inputs": "2\n1\nThis is a short sample sentence for Project Axiom training.\n2\nq\n",
        "expected": [
            "--- Training Mode: ON ---",
            "[..] Parsing and tokenizing multimodal data...",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "6. Training: Text + Image (Photo.jpeg)",
        "inputs": "2\n1\nThis text is accompanied by an image asset.\n1\n1\nPhoto.jpeg\nq\n",
        "expected": [
            "Select Media Type:",
            "Found in",
            "Photo.jpeg",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "7. Training: Text + Video (Video.mp4)",
        "inputs": "2\n1\nThis text is accompanied by a video file.\n1\n2\nVideo.mp4\nq\n",
        "expected": [
            "Select Media Type:",
            "Found in",
            "Video.mp4",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "8. Training: Text + Voice (Voice.ogg)",
        "inputs": "2\n1\nThis text is accompanied by audio voice data.\n1\n3\nVoice.ogg\nq\n",
        "expected": [
            "Select Media Type:",
            "Found in",
            "Voice.ogg",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "9. Training: Number (No Media)",
        "inputs": "2\n2\n1.2 3.4 5.6 7.8 9.0\n2\nq\n",
        "expected": [
            "Enter Number:",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "10. Training: Number + Image (Photo.jpeg)",
        "inputs": "2\n2\n42.0 99.5 101.2\n1\n1\nPhoto.jpeg\nq\n",
        "expected": [
            "Select Media Type:",
            "Found in",
            "Photo.jpeg",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "11. Training: Image Only (Photo.jpeg)",
        "inputs": "2\n3\nPhoto.jpeg\nq\n",
        "expected": [
            "Enter Image Path:",
            "Found in",
            "Photo.jpeg",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "12. Training: Video Only (Video.mp4)",
        "inputs": "2\n4\nVideo.mp4\nq\n",
        "expected": [
            "Enter Video Path:",
            "Found in",
            "Video.mp4",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    },
    {
        "name": "13. Training: Voice Only (Voice.ogg)",
        "inputs": "2\n5\nVoice.ogg\nq\n",
        "expected": [
            "Enter Voice Path:",
            "Found in",
            "Voice.ogg",
            "Vocabulary prepared:",
            "Beginning continuous training loop...",
            "[OK] Checkpoint saved. Exiting training."
        ]
    }
]

total_passed = 0
total_failed = 0

print("=" * 60)
print("  EXECUTING COMPREHENSIVE INTERACTIVE SCENARIO TEST SUITE")
print("=" * 60)

for sc in scenarios:
    name = sc["name"]
    inputs = sc["inputs"]
    expected = sc["expected"]
    
    print(f"\n[RUNNING] {name}...", flush=True)
    p = subprocess.Popen(
        ['app.exe'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    out, err = "", ""
    try:
        out, err = p.communicate(input=inputs, timeout=25)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        print("  [ERROR] Execution timed out!")
    except Exception as e:
        p.kill()
        out, err = p.communicate()
        print(f"  [ERROR] Subprocess exception: {e}")
        
    combined = out + "\n" + err
    success = True
    for marker in expected:
        if marker not in combined:
            print(f"  [FAIL] Missing expected marker: \"{marker}\"")
            success = False
            
    if success:
        print(f"  [PASS] {name} executed successfully!", flush=True)
        total_passed += 1
    else:
        print(f"  [FAIL] {name} failed validation.", flush=True)
        total_failed += 1
        print("--- Standard Output ---")
        print(out[:1000])
        print("--- Standard Error ---")
        print(err[:500])

print("\n" + "=" * 60)
print(f"INTERACTIVE SCENARIO SUMMARY: {total_passed} PASSED, {total_failed} FAILED")
print("=" * 60)

if total_failed > 0:
    sys.exit(1)
sys.exit(0)
