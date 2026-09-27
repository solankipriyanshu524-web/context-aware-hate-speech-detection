import os
import io
import sys
import subprocess
import tempfile
import numpy as np
import librosa
import soundfile as sf
import imageio_ffmpeg
from src.predict import HateSpeechPredictor

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def setup_cuda_dlls():
    """Ensure Windows finds NVIDIA CUDA & cuDNN DLLs installed via pip."""
    if sys.platform == "win32":
        site_pkgs = os.path.join(sys.prefix, 'Lib', 'site-packages')
        nvidia_path = os.path.join(site_pkgs, 'nvidia')
        if os.path.isdir(nvidia_path):
            for root, dirs, _ in os.walk(nvidia_path):
                if 'bin' in dirs:
                    bin_dir = os.path.join(root, 'bin')
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
                    if bin_dir not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')

class VoiceHateSpeechDetector:
    def __init__(self, models_dir='models', whisper_model_size='base'):
        self.predictor = HateSpeechPredictor(models_dir=models_dir)
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.whisper_model_size = whisper_model_size
        self._whisper = None
        self.device_used = "cpu"

    @property
    def whisper(self):
        """Lazy load Whisper model on CUDA GPU with fast CPU int8 fallback."""
        if self._whisper is None:
            setup_cuda_dlls()
            try:
                from faster_whisper import WhisperModel
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    try:
                        self._whisper = WhisperModel(self.whisper_model_size, device="cuda", compute_type="float16")
                        self.device_used = "cuda"
                    except Exception as cuda_err:
                        print(f"⚠️ CUDA load fallback to CPU: {cuda_err}")
                        self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=8)
                        self.device_used = "cpu"
                else:
                    self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=8)
                    self.device_used = "cpu"
            except Exception as e:
                print(f"⚠️ Whisper load failed: {e}")
                self._whisper = None
        return self._whisper

    def analyze_acoustic_tone(self, y, sample_rate=16000):
        """
        Analyzes the acoustic energy and frequency variation of the voice signal.
        Determines if the tone is aggressive/shouting, normal, or calm.
        """
        try:
            rms = float(np.mean(librosa.feature.rms(y=y)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
            
            rms_norm = round(min(1.0, rms * 10), 2)
            zcr_norm = round(min(1.0, zcr * 5), 2)
            
            if rms > 0.08 or zcr > 0.15:
                tone = "🔥 Aggressive / Shouting Tone Detected"
                tone_score = 0.80
            elif rms > 0.02:
                tone = "⚡ Moderate / Conversational Tone"
                tone_score = 0.40
            else:
                tone = "🍃 Calm / Low Volume Tone"
                tone_score = 0.15
                
            return {
                'tone': tone,
                'tone_score': tone_score,
                'rms': rms_norm,
                'zcr': zcr_norm
            }
        except Exception:
            return {
                'tone': "⚡ Conversational Tone",
                'tone_score': 0.30,
                'rms': 0.05,
                'zcr': 0.05
            }

    def convert_to_wav(self, audio_source):
        """
        Converts any audio format (.m4a, .mp3, .wav, .ogg, file buffer)
        into a clean 16kHz WAV format using static ffmpeg.
        """
        temp_input = None
        temp_wav = None

        if hasattr(audio_source, 'read'):
            audio_bytes = audio_source.read()
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.audio')
            temp_input.write(audio_bytes)
            temp_input.flush()
            temp_input.close()
            source_path = temp_input.name
        else:
            source_path = str(audio_source)

        try:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()

            # Execute ffmpeg to convert to 16kHz mono 16-bit PCM WAV
            cmd = [
                self.ffmpeg_exe,
                '-y',
                '-i', source_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                temp_wav_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return None, None, 16000, 0.0, "FFmpeg decoding failed."

            # Read decoded audio with soundfile
            y, sr_rate = sf.read(temp_wav_path)
            y = np.ascontiguousarray(y, dtype=np.float32)
            duration = len(y) / float(sr_rate) if sr_rate > 0 else 0.0
            
            return temp_wav_path, y, sr_rate, duration, None

        except Exception as e:
            return None, None, 16000, 0.0, str(e)
        finally:
            if temp_input and os.path.exists(temp_input.name):
                try:
                    os.remove(temp_input.name)
                except Exception:
                    pass

    def transcribe(self, wav_path):
        """
        Transcribes speech using fast multi-tier ASR (Google Online -> Offline Whisper fallback).
        """
        # Tier 1: Fast Google Online Speech Recognition (~1-2 seconds)
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                
                # hi-IN handles Hindi, Hinglish, and English accents seamlessly
                for lang in ["hi-IN", "en-IN"]:
                    try:
                        text = recognizer.recognize_google(audio_data, language=lang)
                        if text and text.strip():
                            return text.strip(), lang, None
                    except Exception:
                        continue
        except Exception:
            pass

        # Tier 2: Offline Faster-Whisper Model
        if self.whisper is not None:
            try:
                segments, info = self.whisper.transcribe(wav_path, beam_size=1, vad_filter=True)
                text_list = [seg.text.strip() for seg in segments]
                full_text = " ".join(text_list).strip()
                if full_text:
                    detected_lang = f"{info.language}"
                    return full_text, detected_lang, None
            except Exception:
                pass

        return None, None, "No speech detected in audio."

    def analyze(self, audio_source):
        """
        Full End-to-End Voice Hate Speech Detection Pipeline:
        Audio -> WAV Normalize -> Acoustic Prosody -> Transcription -> NLP Model + Context Check
        """
        # Step 1: Normalize Audio
        wav_path, y_audio, sr_rate, duration, err = self.convert_to_wav(audio_source)
        if err or wav_path is None:
            return {
                'success': False,
                'error': f"Audio error: {err}"
            }

        try:
            if duration < 0.2:
                return {
                    'success': False,
                    'error': "Audio is too short or silent."
                }

            # Step 2: Acoustic Tone Analysis
            acoustic_info = self.analyze_acoustic_tone(y_audio, sr_rate)

            # Step 3: Transcription
            transcription, detected_lang, trans_err = self.transcribe(wav_path)
            if trans_err or not transcription:
                return {
                    'success': False,
                    'error': trans_err or "Could not transcribe spoken words.",
                    'acoustic_info': acoustic_info,
                    'duration': round(duration, 2)
                }

            # Step 4: NLP Hate Speech & Context Prediction
            prediction_result = self.predictor.predict(transcription)

            return {
                'success': True,
                'transcription': transcription,
                'detected_lang': detected_lang,
                'duration': round(duration, 2),
                'acoustic_info': acoustic_info,
                'prediction': prediction_result['prediction'],
                'cleaned_text': prediction_result['cleaned_text'],
                'context': prediction_result['context']
            }
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

if __name__ == "__main__":
    detector = VoiceHateSpeechDetector(whisper_model_size='base')
    test_audio = os.path.join("..", "voice.m4a")
    if os.path.exists(test_audio):
        print(f"Testing Voice Hate Speech Detector on: {test_audio}")
        res = detector.analyze(test_audio)
        print("\n--- VOICE DETECTION REPORT ---")
        if res.get('success'):
            print(f"🗣️ Transcribed Text : {res['transcription']}")
            print(f"🌐 Language         : {res['detected_lang']}")
            print(f"⏱️ Duration         : {res['duration']}s")
            print(f"🔊 Acoustic Tone    : {res['acoustic_info']['tone']}")
            print(f"⚖️ Verdict          : {res['prediction'].upper()}")
            if res.get('context'):
                print(f"🔮 Context Reason   : {res['context']['reason']}")
        else:
            print(f"❌ Error: {res.get('error')}")
    else:
        print("Detector initialized successfully. Ready for voice streams!")
