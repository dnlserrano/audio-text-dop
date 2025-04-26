from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import subprocess
import math
import os
import re
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any

class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "large",
        language: str = "pt",
        gpu_backend: str = "auto",
        chunk_duration: int = 30
    ):
        self.model_size = model_size
        self.language = language
        self.chunk_duration = chunk_duration
        self.device = self._setup_device(gpu_backend)

        # Lazy loading - will only load model when needed
        self.model = None
        self.processor = None

    def _setup_device(self, gpu_backend: str) -> torch.device:
        """Set up the compute device based on user preference and availability"""
        if gpu_backend == "auto":
            # Automatically detect the best available option
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using CUDA GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using Apple M1/M2 GPU (MPS)")
            else:
                device = torch.device("cpu")
                print("No GPU found, using CPU")
        elif gpu_backend == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        elif gpu_backend == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple M1/M2 GPU (MPS)")
        else:
            if gpu_backend != "cpu":
                print(f"Warning: Requested GPU backend '{gpu_backend}' not available. Falling back to CPU.")
            device = torch.device("cpu")
            print("Using CPU")

        return device

    def _load_model(self):
        """Lazy load the model only when needed"""
        if self.model is None or self.processor is None:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{self.model_size}")
            self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            self.model = self.model.to(self.device)
            print(f"Model loaded and moved to {self.device}")

    def remove_silence_and_convert(self, input_file: str, output_file: str) -> str:
        """Remove silence from audio and convert to 16kHz WAV format"""
        # First create a temporary file for the silence-removed version
        temp_file = f"temp_{os.path.basename(input_file)}_no_silence.mp4"

        # Remove silence - slightly adjusted parameters for Portuguese speech patterns
        silence_remove_cmd = [
            "ffmpeg", "-i", input_file,
            "-af", "silenceremove=stop_periods=-1:stop_duration=0.8:stop_threshold=-45dB",
            "-y", temp_file
        ]
        subprocess.run(silence_remove_cmd, check=True)

        # Convert to WAV with 16kHz sample rate
        convert_cmd = [
            "ffmpeg", "-i", temp_file,
            "-ac", "1", "-ar", "16000",
            "-y", output_file
        ]
        subprocess.run(convert_cmd, check=True)

        # Remove temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return output_file

    def get_audio_duration(self, file_path: str) -> float:
        info = torchaudio.info(file_path)
        duration_seconds = info.num_frames / info.sample_rate
        return duration_seconds

    def load_audio_chunk(self, file_path: str, start_time: Any, target_sample_rate: int = 16000) -> tuple:
        # Load full audio
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert start_time from MM:SS to seconds
        if isinstance(start_time, str) and ":" in start_time:
            minutes, seconds = map(int, start_time.split(":"))
            start_seconds = minutes * 60 + seconds
        else:
            start_seconds = float(start_time)

        # Calculate start and end samples
        start_sample = int(start_seconds * sample_rate)
        end_sample = min(start_sample + int(self.chunk_duration * sample_rate), waveform.shape[1])

        # Extract chunk
        chunk = waveform[:, start_sample:end_sample]

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            chunk = resampler(chunk)

        return chunk, target_sample_rate

    def transcribe_chunk(self, waveform, sample_rate) -> str:
        self._load_model()  # Ensure model is loaded

        # Process audio with specified language
        inputs = self.processor(
            waveform[0].numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move input to the appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Generate with language-specific settings
            generated_ids = self.model.generate(
                inputs["input_features"],
                language=self.language,
                task="transcribe"
            )

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Apply language-specific post-processing
        transcription = self._fix_common_errors(transcription)

        return transcription

    def _fix_common_errors(self, text: str) -> str:
        """Fix common transcription errors in the specified language"""
        # Portuguese-specific corrections if language is Portuguese
        if self.language == "pt":
            corrections = {
                # Common accent errors
                r'\bque\b': 'quê',
                r'\be\b': 'é',
                r'\ba\b': 'à',
                r'\bpor que\b': 'porque',

                # Fix common word confusions for European Portuguese
                r'\bcomo\b': 'como',
                r'\btem\b': 'têm',
                r'\besta\b': 'está',

                # Common punctuation errors
                r'([a-zA-Z])\s+([,.;:!?])': r'\1\2',
            }

            for pattern, replacement in corrections.items():
                text = re.sub(pattern, replacement, text)

        # General punctuation and spacing corrections for all languages
        text = re.sub(r'\s+', ' ', text)  # Fix multiple spaces
        text = text.strip()

        return text

    def is_end_of_sentence(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation"""
        return bool(re.search(r'[.!?]\s*$', text))

    async def transcribe_stream(self, file_path: str, start_time: str = "0:00") -> AsyncGenerator[str, None]:
        """Stream transcription results as they are processed"""
        # Ensure model is loaded
        self._load_model()

        # Process audio file
        wav_file = f"temp_{os.path.basename(file_path)}_processed.wav"
        print(f"Processing {file_path}...")
        print("Removing silence and converting to WAV format...")
        self.remove_silence_and_convert(file_path, wav_file)

        # Get total duration
        total_duration = self.get_audio_duration(wav_file)

        # Convert start_time from MM:SS to seconds
        if isinstance(start_time, str) and ":" in start_time:
            minutes, seconds = map(int, start_time.split(":"))
            start_seconds = minutes * 60 + seconds
        else:
            start_seconds = float(start_time)

        # Calculate number of chunks
        remaining_duration = total_duration - start_seconds
        num_chunks = math.ceil(remaining_duration / self.chunk_duration)

        print(f"Total audio duration: {total_duration:.2f}s")
        print(f"Processing {num_chunks} chunks starting from {start_time}...")

        current_paragraph = ""

        for i in range(num_chunks):
            chunk_start = start_seconds + (i * self.chunk_duration)
            print(f"Processing chunk {i+1}/{num_chunks} starting at {chunk_start:.2f}s...")

            chunk, sample_rate = self.load_audio_chunk(wav_file, chunk_start)

            # Skip processing if chunk is too short or silent
            if chunk.shape[1] < 0.5 * sample_rate:  # Less than 0.5 seconds
                continue

            # Check if chunk is mostly silence
            if torch.abs(chunk).mean() < 0.01:
                print(f"Chunk {i+1} appears to be silence, skipping...")
                continue

            chunk_transcription = self.transcribe_chunk(chunk, sample_rate)
            print(f"Chunk {i+1} transcription: {chunk_transcription}")

            # Add to current paragraph and check if we should end paragraph
            current_paragraph += " " + chunk_transcription.strip()
            current_paragraph = current_paragraph.strip()

            # If we have a reasonably sized paragraph and it ends with sentence-ending punctuation
            # Adjusted threshold based on language characteristics
            min_paragraph_size = 150 if self.language == "pt" else 100

            if len(current_paragraph) > min_paragraph_size and self.is_end_of_sentence(current_paragraph):
                # Yield the paragraph for streaming
                yield current_paragraph
                current_paragraph = ""

            # Allow other tasks to run between chunk processing
            await asyncio.sleep(0)

        # Yield any remaining text as final paragraph
        if current_paragraph.strip():
            yield current_paragraph.strip()

        # Clean up temporary file
        if os.path.exists(wav_file):
            os.remove(wav_file)
            print(f"Removed temporary file: {wav_file}")
