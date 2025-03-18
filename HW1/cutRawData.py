import os
import random
from pydub import AudioSegment
from settings import sampleRate  
import stat
import shutil
# Define pairs: (class name for reference, class label as integer)
pairs = [
    ('C', 0),
    ('D', 1),
    ('E', 2),
    ('F', 3),
    ('G', 4),
    ('A', 5)
]

# Parameters: segment duration in seconds and train ratio
SEGMENT_DURATION = 2  # seconds; change as needed
TRAIN_RATIO = 0.8     # 80% training, 20% test

def split_audio_by_duration(audio: AudioSegment, segment_duration: int):
    """
    Splits the given AudioSegment into segments of fixed duration (in seconds).
    If the last segment is shorter than the segment duration, it is discarded.
    
    Args:
        audio (AudioSegment): The input audio.
        segment_duration (int): Duration in seconds for each segment.
        
    Returns:
        List[AudioSegment]: List of audio segments.
    """
    segment_ms = segment_duration * 1000  # convert to milliseconds
    total_ms = len(audio)
    segments = []
    
    # Loop over the audio in increments of segment_ms
    for start in range(0, total_ms, segment_ms):
        end = start + segment_ms
        if end <= total_ms:
            segments.append(audio[start:end])
    return segments

def remove_readonly(func, path, exc_info):
    # Clear the readonly bit and reattempt the removal.
    os.chmod(path, stat.S_IWRITE)
    func(path)

def ensure_directories(base_dir, labels):
    """
    Creates the output directory structure:
    base_dir/train/<label> and base_dir/test/<label> for each label in labels.
    If a directory already exists, it will be cleared.
    """
    for subset in ['train', 'test']:
        for label in labels:
            dir_path = os.path.join(base_dir, subset, str(label))
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, onerror=remove_readonly)
            os.makedirs(dir_path, exist_ok=True)

def process_class(pair, segment_duration, train_ratio, output_base="dataset"):
    """
    Process a single class file: split into segments and assign each segment
    to train or test directories.
    
    Args:
        pair (tuple): (class_name, class_label)
        segment_duration (int): Duration (seconds) for each segment.
        train_ratio (float): Ratio for training samples.
        output_base (str): Base output directory.
    """
    class_name, label = pair
    input_file = f"dataset/raw/{class_name}.m4a"
    print(f"Processing {input_file} for class {label}...")
    
    # Load the .m4a file (ffmpeg must be installed)
    audio = AudioSegment.from_file(input_file, format="m4a")
    
    # Optionally, convert to desired sample rate if needed using set_frame_rate
    audio = audio.set_frame_rate(sampleRate)
    
    # Split into fixed-duration segments and shuffle it
    segments = split_audio_by_duration(audio, segment_duration)[1:-1]
    random.shuffle(segments)

    num_segments = len(segments)   
    train_size = int(num_segments * train_ratio)
    print(f"Total segments generated: {num_segments}")

    # Save each segment to train or test directory
    for i, segment in enumerate(segments):
        # Randomly assign to train or test based on train_ratio
        if i < train_size:
            subset = "train"
        else:
            subset = "test"
            
        # Build the output file path: dataset/<subset>/<label>/chord_{i}.wav
        output_dir = os.path.join(output_base, subset, str(label))
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"chord_{i}.wav")
        
        # Export segment with specified sample rate
        segment.export(out_path, format="wav", parameters=["-ar", f"{sampleRate}"])
        print(f"Exported: {out_path}")

def main():
    # Define output base directory and ensure train/test subfolders for each class exist
    output_base = "dataset"
    labels = [pair[1] for pair in pairs]
    ensure_directories(output_base, labels)
    
    # Process each class file
    for pair in pairs:
        process_class(pair, SEGMENT_DURATION, TRAIN_RATIO, output_base=output_base)

if __name__ == "__main__":
    main()
