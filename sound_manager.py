import pygame
import os

class SoundManager:
    def __init__(self):
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists('sounds'):
            os.makedirs('sounds')
        
        # Load sound effects (can be either .wav or .mp3)
        self.sounds = {}
        sound_files = {
            'bomb_place': ['bomb_place.wav', 'bomb_place.mp3'],
            'explosion': ['explosion.wav', 'explosion.mp3'],
            'powerup': ['powerup.wav', 'powerup.mp3'],
            'death': ['death.wav', 'death.mp3'],
            'menu_select': ['menu_select.wav', 'menu_select.mp3'],
            'menu_move': ['menu_move.wav', 'menu_move.mp3'],
            'victory': ['victory.wav', 'victory.mp3']
        }
        
        # Try to load each sound file, first try .wav, then .mp3
        for sound_name, file_names in sound_files.items():
            for file_name in file_names:
                file_path = os.path.join('sounds', file_name)
                if os.path.exists(file_path):
                    try:
                        self.sounds[sound_name] = pygame.mixer.Sound(file_path)
                        break
                    except:
                        continue
        
        # Load background music (prefer .mp3)
        music_files = ['background_music.mp3', 'background_music.wav']
        for music_file in music_files:
            music_path = os.path.join('sounds', music_file)
            if os.path.exists(music_path):
                try:
                    pygame.mixer.music.load(music_path)
                    break
                except:
                    continue
        
        # Set volume
        self.set_volume(0.5)
        
    def play_sound(self, sound_name):
        """Play a sound effect"""
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
            
    def play_music(self, loop=-1):
        """Play background music (-1 for infinite loop)"""
        pygame.mixer.music.play(loop)
        
    def stop_music(self):
        """Stop background music"""
        pygame.mixer.music.stop()
        
    def pause_music(self):
        """Pause background music"""
        pygame.mixer.music.pause()
        
    def unpause_music(self):
        """Unpause background music"""
        pygame.mixer.music.unpause()
        
    def set_volume(self, volume):
        """Set volume for all sounds (0.0 to 1.0)"""
        for sound in self.sounds.values():
            sound.set_volume(volume)
        pygame.mixer.music.set_volume(volume)
        
    def add_sound(self, name, file_path):
        """Add a new sound effect"""
        if os.path.exists(file_path):
            try:
                self.sounds[name] = pygame.mixer.Sound(file_path)
                return True
            except:
                return False
        return False 