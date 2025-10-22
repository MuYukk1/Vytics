#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Video Understanding System
A simple script to process videos by extracting frames and audio,
then analyzing them using DashScope API models.
"""

import os
import json
import time
import base64
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
from moviepy.editor import VideoFileClip
import tempfile
import shutil
from PIL import Image
import io
import dashscope
from openai import OpenAI
from pathlib import Path

# Load environment variables
load_dotenv()

class VideoAnalyzer:
    def __init__(self):
        """Initialize the video analyzer with configuration from .env file"""
        self.api_key = os.getenv('API_key')
        self.base_url = os.getenv('base_url')
        self.vl_model = os.getenv('VL_ModelName', 'qwen3-vl-plus-2025-09-23')
        self.text_model = os.getenv('Text_ModelName', 'qwen3-max-2025-09-23')
        self.asr_model = os.getenv('ASR_ModelName', 'qwen3-omni-flash')
        self.frame_per_second = float(os.getenv('frame_per_second', 0.1))  # Use 0.1 fps (24 frames for 4min video)
        self.max_frame = int(os.getenv('max_frame', 5))  # Maximum number of frames to analyze
        self.audio_duration = int(os.getenv('audio_duration', 60))  # Maximum audio duration in seconds
        
        # Initialize token usage tracking
        self.token_usage = {
            'vl_model': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'asr_model': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'text_model': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'total': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        }
        
        # Validate configuration
        if not self.api_key or not self.base_url:
            raise ValueError("API_key and base_url must be set in .env file")
        
        # Set DashScope API key for SDK
        dashscope.api_key = self.api_key
        
        print(f"Initialized VideoAnalyzer with:")
        print(f"  VL Model: {self.vl_model}")
        print(f"  Text Model: {self.text_model}")
        print(f"  ASR Model: {self.asr_model}")
        print(f"  Frame Rate: {self.frame_per_second} fps")
        print(f"  Max Frames: {self.max_frame}")
        print(f"  Audio Duration: {self.audio_duration}s")

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Main function to process a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"\nStarting video analysis for: {video_path}")
        start_time = time.time()
        frames = []
        audio_path = None
        
        try:
            # Step 1: Extract frames and audio
            print("Step 1: Extracting frames and audio...")
            frames, audio_path = self.extract_frames_and_audio(video_path)
            
            # Step 2: Analyze with DashScope API
            print("Step 2: Analyzing with AI models...")
            results = self.analyze_with_dashscope(frames, audio_path)
            
            # Step 3: Format and save results
            processing_time = time.time() - start_time
            
            # Calculate total token usage
            self.calculate_total_token_usage()
            
            results['processing_time'] = processing_time
            results['video_path'] = video_path
            results['frames_extracted'] = len(frames)
            results['has_audio'] = audio_path is not None
            results['token_usage'] = self.token_usage
            
            print(f"Analysis completed in {processing_time:.2f} seconds")
            self.print_token_usage_summary()
            return results
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            print("Cleaning up temporary files...")
            self.cleanup_temp_files(frames, audio_path)

    def extract_frames_and_audio(self, video_path: str) -> tuple:
        """
        Extract frames and audio from video file using moviepy
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames_list, audio_file_path)
        """
        print(f"  Extracting frames at {self.frame_per_second} fps...")
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")
        frames = []
        audio_path = None
        
        try:
            # Load video using moviepy
            video_clip = VideoFileClip(video_path)
            duration = video_clip.duration
            fps = video_clip.fps
            
            print(f"  Video info: {duration:.2f}s, {fps:.2f} fps")
            
            # Calculate time intervals for frame extraction
            frame_interval = 1.0 / self.frame_per_second
            extracted_count = 0
            
            # Extract frames at specified intervals
            current_time = 0
            while current_time < duration:
                try:
                    # Get frame at current time
                    frame = video_clip.get_frame(current_time)
                    
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # Save frame
                    frame_filename = f"frame_{extracted_count:04d}_{current_time:.2f}s.jpg"
                    frame_path = os.path.join(temp_dir, frame_filename)
                    pil_image.save(frame_path, 'JPEG', quality=85)
                    
                    # Convert frame to base64 for API
                    with open(frame_path, 'rb') as f:
                        frame_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    frames.append({
                        'timestamp': current_time,
                        'image_path': frame_path,
                        'image_base64': frame_base64,
                        'sequence': extracted_count
                    })
                    
                    extracted_count += 1
                    current_time += frame_interval
                    
                except Exception as e:
                    print(f"  Warning: Failed to extract frame at {current_time:.2f}s: {str(e)}")
                    current_time += frame_interval
                    continue
            
            print(f"  Extracted {extracted_count} frames from {duration:.1f}s video")
            
            # Extract audio
            print("  Extracting audio...")
            try:
                if video_clip.audio is not None:
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                    print(f"  Audio extracted to: {audio_path}")
                else:
                    print("  No audio track found in video")
            except Exception as e:
                print(f"  Audio extraction failed: {str(e)}")
                audio_path = None
            
            # Close video clip
            video_clip.close()
            
            return frames, audio_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e

    def analyze_with_dashscope(self, frames: List, audio_path: str) -> Dict[str, Any]:
        """
        Send data to DashScope API for analysis
        
        Args:
            frames: List of frame data
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'visual_analysis': '',
            'audio_transcription': '',
            'final_summary': ''
        }
        
        try:
            # Step 1: Analyze frames with VL model
            print("  Analyzing frames with VL model...")
            visual_analysis = self.analyze_frames_with_vl_model(frames)
            results['visual_analysis'] = visual_analysis
            
            # Step 2: Transcribe audio with ASR model (if available)
            print("  Transcribing audio with ASR model...")
            if audio_path:
                audio_transcription = self.transcribe_audio_with_asr(audio_path)
                results['audio_transcription'] = audio_transcription
            else:
                results['audio_transcription'] = "No audio available"
            
            # Step 3: Generate final summary with text model
            print("  Generating final summary with text model...")
            final_summary = self.generate_summary_with_text_model(
                visual_analysis, results['audio_transcription']
            )
            results['final_summary'] = final_summary
            
            return results
            
        except Exception as e:
            print(f"  Error in DashScope API analysis: {str(e)}")
            return {
                'visual_analysis': f'Error: {str(e)}',
                'audio_transcription': f'Error: {str(e)}',
                'final_summary': f'Error: {str(e)}'
            }

    def analyze_frames_with_vl_model(self, frames: List) -> str:
        """
        Analyze frames using Vision-Language model
        
        Args:
            frames: List of frame data with base64 images
            
        Returns:
            String containing visual analysis
        """
        if not frames:
            return "No frames to analyze"
        
        # Prepare messages for VL model - analyze multiple frames
        max_frames = min(self.max_frame, len(frames))  # Use configurable max_frame
        # Select frames evenly distributed across the video
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            selected_frames = [frames[i * step] for i in range(max_frames)]
        else:
            selected_frames = frames
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """你是一名视频分析师，擅长对各种视频片段进行理解。

请分析这些视频关键帧图片，完成以下任务：
- 描述每张图片的画面信息，包括人物、物体、动作、文字、字幕、镜头语言等
- 将所有图片信息串联起来，生成视频的详细概述，还原该片段的剧情

请按时间顺序分析每一帧，然后给出整体的视频内容描述。"""
                }
            ]
        }]
        
        # Add frame images to the message
        print(f"    Analyzing {len(selected_frames)} frames...")
        for i, frame in enumerate(selected_frames):
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['image_base64']}"
                }
            })
            print(f"    Added frame {i+1}/{len(selected_frames)} (timestamp: {frame['timestamp']:.1f}s)")
        
        return self.call_dashscope_api(self.vl_model, messages)

    def transcribe_audio_with_asr(self, audio_path: str) -> str:
        """
        Transcribe audio using OpenAI-compatible API with qwen3-omni-flash model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            String containing transcription
        """
        try:
            # Create a shorter audio clip (first 30 seconds) to reduce API load
            temp_dir = os.path.dirname(audio_path)
            short_audio_path = os.path.join(temp_dir, "audio_short.mp3")
            
            # Use moviepy to create a shorter audio clip
            from moviepy.editor import AudioFileClip
            audio_clip = AudioFileClip(audio_path)
            # Use configurable audio duration
            max_duration = min(self.audio_duration, audio_clip.duration)
            short_clip = audio_clip.subclip(0, max_duration)
            print(f"    Processing {max_duration:.1f}s of {audio_clip.duration:.1f}s total audio")
            # Write audio with aggressive compression to reduce file size
            short_clip.write_audiofile(short_audio_path, verbose=False, logger=None, 
                                     bitrate="32k",      # Very low bitrate for speech
                                     fps=8000,           # Low sample rate for speech
                                     codec='mp3')        # Use MP3 for better compression
            short_clip.close()
            audio_clip.close()
            
            print(f"    Transcribing audio file: {short_audio_path}")
            
            # Get file format
            file_format = Path(short_audio_path).suffix.lstrip('.')
            mime_type_map = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "m4a": "audio/x-m4a",
                "flac": "audio/flac",
                "aac": "audio/aac",
                "ogg": "audio/ogg",
            }
            mime_type = mime_type_map.get(file_format.lower(), f"audio/{file_format}")
            
            # Initialize OpenAI client for DashScope
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            
            # Read audio file and encode to base64
            with open(short_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            print(f"    Audio file size: {len(audio_bytes) / 1024:.1f} KB")
            
            # Create transcription request with retry
            max_retries = 3
            completion = None
            
            for attempt in range(max_retries):
                try:
                    print(f"    Attempting transcription (attempt {attempt + 1}/{max_retries})...")
                    completion = client.chat.completions.create(
                        model=self.asr_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": f"data:{mime_type};base64,{audio_b64}",
                                            "format": file_format,
                                        },
                                    },
                                    {"type": "text", "text": "请转录这段音频内容"},
                                ],
                            },
                        ],
                        modalities=["text"],
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    break
                except Exception as inner_err:
                    msg = str(inner_err)
                    is_rate_or_unavailable = (
                        "Too many requests" in msg or "ServiceUnavailable" in msg or "throttled" in msg
                    )
                    if attempt < max_retries - 1 and is_rate_or_unavailable:
                        wait = 2 ** attempt
                        print(f"    Rate limited, retrying in {wait}s... (attempt {attempt+1})")
                        time.sleep(wait)
                        continue
                    raise inner_err
            
            if completion is None:
                raise Exception("Failed to create transcription request")
            
            # Collect transcription result
            transcript = ""
            total_tokens = 0
            
            for chunk in completion:
                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        transcript += chunk.choices[0].delta.content
                
                # Track token usage if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                    if hasattr(usage, 'total_tokens'):
                        total_tokens = usage.total_tokens
                        self.token_usage['asr_model']['total_tokens'] = total_tokens
                        if hasattr(usage, 'prompt_tokens'):
                            self.token_usage['asr_model']['prompt_tokens'] = usage.prompt_tokens
                        if hasattr(usage, 'completion_tokens'):
                            self.token_usage['asr_model']['completion_tokens'] = usage.completion_tokens
            
            # Clean up temporary short audio file
            if os.path.exists(short_audio_path):
                os.remove(short_audio_path)
            
            transcript = transcript.strip()
            if transcript:
                print(f"    ASR transcription completed: {len(transcript)} characters")
                if total_tokens > 0:
                    print(f"    ASR tokens used: {total_tokens}")
                return transcript
            else:
                return "ASR returned empty transcription"
            
        except Exception as e:
            print(f"  ASR transcription failed: {str(e)}")
            # Clean up on error
            if 'short_audio_path' in locals() and os.path.exists(short_audio_path):
                os.remove(short_audio_path)
            return f"Audio transcription failed: {str(e)}"

    def generate_summary_with_text_model(self, visual_analysis: str, audio_transcription: str) -> str:
        """
        Generate final summary using text model
        
        Args:
            visual_analysis: Visual analysis from VL model
            audio_transcription: Audio transcription from ASR model
            
        Returns:
            String containing final summary
        """
        prompt = f"""你是一个专业的视频标注专员，擅长结合视频镜头信息来分析处理各种视频任务。

请你结合以下输入数据，串联、还原出整个视频的详细剧情：

## 视频分镜信息（视频各镜头的视觉描述信息）：
{visual_analysis}

## 视频ASR转录信息（音频转录文字）：
{audio_transcription}

请根据以上信息：
1. 如出现语法错误或逻辑不通，请直接修改
2. 根据剧情进展，准确判断每段台词的真实说话者
3. 如果视频分镜中无台词，请根据音频文字为其匹配台词
4. 适当保留视频分镜中对人物、场景的描写
5. 润色故事，使其更具逻辑性
6. 统一不同分镜中的人物角色

请直接输出视频剧情，不要输出其他信息。"""

        messages = [{
            "role": "user",
            "content": prompt
        }]
        
        return self.call_dashscope_api(self.text_model, messages)

    def call_dashscope_api(self, model: str, messages: List) -> str:
        """
        Call DashScope API with retry logic and token usage tracking
        
        Args:
            model: Model name to use
            messages: Messages for the API call
            
        Returns:
            String response from the API
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        # Track token usage
                        if 'usage' in result:
                            usage = result['usage']
                            model_type = 'vl_model' if 'vl' in model else 'text_model'
                            
                            prompt_tokens = usage.get('prompt_tokens', 0)
                            completion_tokens = usage.get('completion_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0)
                            
                            self.token_usage[model_type]['prompt_tokens'] += prompt_tokens
                            self.token_usage[model_type]['completion_tokens'] += completion_tokens
                            self.token_usage[model_type]['total_tokens'] += total_tokens
                            
                            print(f"    {model_type} tokens used: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
                        
                        return result['choices'][0]['message']['content']
                    else:
                        return f"API returned empty response: {result}"
                else:
                    error_msg = f"API call failed with status {response.status_code}: {response.text}"
                    print(f"  Attempt {attempt + 1} failed: {error_msg}")
                    if attempt == max_retries - 1:
                        return f"API Error: {error_msg}"
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                print(f"  Attempt {attempt + 1} failed: {error_msg}")
                if attempt == max_retries - 1:
                    return f"Request Error: {error_msg}"
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return "Failed to get response after all retries"

    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """
        Save analysis results to file
        
        Args:
            results: Analysis results dictionary
            output_path: Optional output file path
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"video_analysis_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_path}")

    def calculate_total_token_usage(self):
        """Calculate total token usage across all models"""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        
        for model_type in ['vl_model', 'asr_model', 'text_model']:
            total_prompt += self.token_usage[model_type]['prompt_tokens']
            total_completion += self.token_usage[model_type]['completion_tokens']
            total_tokens += self.token_usage[model_type]['total_tokens']
        
        self.token_usage['total'] = {
            'prompt_tokens': total_prompt,
            'completion_tokens': total_completion,
            'total_tokens': total_tokens
        }

    def print_token_usage_summary(self):
        """Print a summary of token usage"""
        print("\n=== Token Usage Summary ===")
        for model_type, usage in self.token_usage.items():
            if model_type != 'total' and usage['total_tokens'] > 0:
                print(f"{model_type}: {usage['total_tokens']} tokens (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
        
        total = self.token_usage['total']
        print(f"Total: {total['total_tokens']} tokens (prompt: {total['prompt_tokens']}, completion: {total['completion_tokens']})")

    def cleanup_temp_files(self, frames: List[Dict], audio_path: str = None):
        """
        Clean up temporary files created during processing
        
        Args:
            frames: List of frame data with file paths
            audio_path: Path to temporary audio file
        """
        try:
            # Clean up frame files
            temp_dirs = set()
            for frame in frames:
                if 'image_path' in frame and os.path.exists(frame['image_path']):
                    temp_dirs.add(os.path.dirname(frame['image_path']))
            
            # Clean up audio file directory
            if audio_path and os.path.exists(audio_path):
                temp_dirs.add(os.path.dirname(audio_path))
            
            # Remove temporary directories
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"  Cleaned up temporary directory: {temp_dir}")
                    
        except Exception as e:
            print(f"  Warning: Failed to clean up temporary files: {str(e)}")


def find_video_files(directory="."):
    """
    Find all video files in the specified directory
    
    Args:
        directory: Directory to search for video files
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    video_files = []
    
    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file.lower())
                if ext in video_extensions:
                    video_files.append(file)
    except Exception as e:
        print(f"Error scanning directory: {str(e)}")
    
    return sorted(video_files)

def select_video_file():
    """
    Let user select a video file from available options
    
    Returns:
        Selected video file path or None if cancelled
    """
    print("Scanning current directory for video files...")
    video_files = find_video_files()
    
    if not video_files:
        print("No video files found in current directory.")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp")
        return None
    
    print(f"\nFound {len(video_files)} video file(s):")
    print("-" * 50)
    
    for i, video_file in enumerate(video_files, 1):
        file_size = os.path.getsize(video_file) / (1024 * 1024)  # Size in MB
        print(f"{i}. {video_file} ({file_size:.1f} MB)")
    
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect a video file (1-{len(video_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Cancelled by user.")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_file = video_files[choice_num - 1]
                print(f"Selected: {selected_file}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(video_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            return None

def main():
    """Main function to run the video analyzer"""
    try:
        print("=== AI Video Understanding System ===")
        
        # Select video file
        video_path = select_video_file()
        if not video_path:
            return 0
        
        # Initialize analyzer
        print("\nInitializing video analyzer...")
        analyzer = VideoAnalyzer()
        
        # Process the selected video
        results = analyzer.process_video(video_path)
        
        # Save and display results
        analyzer.save_results(results)
        
        print("\n=== Analysis Results ===")
        print(f"Video: {results['video_path']}")
        print(f"Processing Time: {results['processing_time']:.2f} seconds")
        print(f"Frames Extracted: {results['frames_extracted']}")
        print(f"Has Audio: {results['has_audio']}")
        
        # Display token usage
        if 'token_usage' in results:
            total_tokens = results['token_usage']['total']['total_tokens']
            print(f"Total Tokens Used: {total_tokens}")
        
        print(f"\nVisual Analysis: {results['visual_analysis'][:200]}...")
        print(f"\nAudio Transcription: {results['audio_transcription']}")  # Show full transcription
        print(f"\nFinal Summary: {results['final_summary'][:200]}...")
        
        print(f"\nFull results saved to JSON file.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())