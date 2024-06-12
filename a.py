import pygame
import os

def play_audio(audio_path):
    # Initialize Pygame Mixer
    pygame.mixer.init()

    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Error: The file {audio_path} does not exist.")
        return

    # Load the audio file
    try:
        pygame.mixer.music.load(audio_path)
        print(f"Playing {audio_path}")
    except pygame.error as e:
        print(f"Failed to load audio file {audio_path}: {e}")
        return

    # Play the audio file
    pygame.mixer.music.play()

    # Wait for the playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

if __name__ == "__main__":
    # Specify the path to your audio file
    audio_file_path = 'C:/Users/Hp/angry_audio.mp3'

    # Call the function to play audio
    play_audio(audio_file_path)
