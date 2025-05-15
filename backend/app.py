from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from datetime import datetime
import httpx
import logging
from typing import Dict, Any, List, Tuple
import json
import concurrent.futures
from functools import partial
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable validation
required_env_vars = [
    'SPOTIFY_CLIENT_ID',
    'SPOTIFY_CLIENT_SECRET',
    'OPENAI_API_KEY',
    'JWT_SECRET'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000", 
            "https://localhost:3000", 
            "http://localhost:19006", 
            "https://localhost:19006", 
            "exp://localhost:19000", 
            "http://localhost:19000",
            "https://*.ngrok.app",
            "https://*.ngrok.io",
            "exp://*-8081.exp.direct",
            "exp://*.*",
            "exp://192.168.*.*:8081",
            "exp://192.168.*.*:19000",
            "exp://127.0.0.1:8081",
            "exp://127.0.0.1:19000"
        ],
        "methods": ["GET", "POST", "OPTIONS", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize OpenAI
client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    http_client=httpx.Client(
        timeout=120.0,  # Increase timeout to 120 seconds for large playlists
        verify=True
    )
)

# Spotify configuration
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = 'musyka://auth'

# Initialize Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=' '.join([
        'playlist-modify-public',
        'playlist-modify-private',
        'user-read-private',
        'user-read-email',
        'user-read-recently-played',
        'user-read-playback-state',
        'user-top-read',
        'user-read-currently-playing',
        'streaming'
    ])
)

def get_spotify_client(token_info: Dict[str, Any]) -> spotipy.Spotify:
    """Initialize and return a Spotify client with the given token."""
    if not token_info or 'access_token' not in token_info:
        raise ValueError("Invalid token_info: access_token is required")
    return spotipy.Spotify(auth=token_info['access_token'])

def generate_playlist_suggestions(prompt: str, song_count: int = 20, existing_tracks: list = None) -> str:
    """Generate playlist suggestions using OpenAI."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create context from existing tracks
            existing_tracks_context = ""
            if existing_tracks and isinstance(existing_tracks, list):
                track_list = []
                for track in existing_tracks[:5]:  # Limit to 5 tracks for context
                    if isinstance(track, dict):
                        # Handle track object format
                        track_name = track.get('name', '')
                        artists = [artist.get('name', '') for artist in track.get('artists', [])]
                        artist_names = ', '.join(artists)
                        if track_name and artist_names:
                            track_list.append(f"• {track_name} by {artist_names}")
                    elif isinstance(track, str):
                        # Handle simple string format
                        track_list.append(f"• {track}")
                
                if track_list:
                    existing_tracks_context = "Some songs already in the playlist:\n" + "\n".join(track_list) + "\n\n"

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": f"""You are a music expert. Generate a list of exactly {song_count} songs that match the user's request and are similar to the existing songs in their playlist.
                    For each song, provide the title and artist in the format 'Title - Artist'.
                    DO NOT number the songs or add any explanations.
                    Focus on musical similarity, style, mood, and era consistency."""},
                    {"role": "user", "content": f"{existing_tracks_context}Create a playlist with {song_count} songs that are similar to the existing songs and match this prompt: {prompt}"}
                ],
                temperature=0.8,
            )
            
            suggestions = response.choices[0].message.content
            
            # Count number of songs generated
            song_lines = [line for line in suggestions.split('\n') if '-' in line and line.strip()]
            num_songs = len(song_lines)
            
            logger.info(f"Generated {num_songs} song suggestions (requested {song_count})")
            
            if num_songs < song_count * 0.8:  # Allow for some flexibility
                logger.warning(f"Too few songs generated ({num_songs}), regenerating")
                return generate_playlist_suggestions(prompt, song_count, existing_tracks)
                
            return suggestions
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Error generating suggestions after {max_retries} attempts: {str(e)}")
                return None
    
    return None  # All retries failed

def validate_track_for_prompt(track_info: dict, prompt: str, sp: spotipy.Spotify, existing_tracks: list = None) -> float:
    """
    Validate if a track matches the given prompt using:
    1. GPT-based semantic matching
    2. Genre matching
    Returns a confidence score between 0-1.
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # First, get GPT's opinion on the match
            song_title = track_info['name']
            artist_name = track_info['artists'][0]['name']
            
            # Create context from existing tracks
            existing_tracks_context = ""
            if existing_tracks and isinstance(existing_tracks, list):
                track_list = []
                for track in existing_tracks[:5]:  # Limit to 5 tracks for context
                    if isinstance(track, dict):
                        # Handle track object format
                        track_name = track.get('name', '')
                        artists = [artist.get('name', '') for artist in track.get('artists', [])]
                        artist_names = ', '.join(artists)
                        if track_name and artist_names:
                            track_list.append(f"• {track_name} by {artist_names}")
                
                if track_list:
                    existing_tracks_context = "Some songs already in the playlist:\n" + "\n".join(track_list) + "\n\n"
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": """You are a music expert. Your task is to evaluate how well a song matches a given prompt and existing playlist.
                    Return ONLY a score between 0 and 1, where:
                    1.0 = Perfect match with both prompt and playlist style
                    0.7-0.9 = Strong match with prompt and similar to playlist
                    0.5-0.6 = Moderate match with either prompt or playlist style
                    0.2-0.4 = Weak match but might fit one aspect
                    0.0-0.1 = No match with prompt or playlist style
                    
                    Consider:
                    - Song title meaning and mood
                    - Artist's typical style
                    - Similarity to existing playlist songs
                    - Genre consistency
                    - Era and production style match
                    
                    Return ONLY the numerical score, nothing else."""},
                    {"role": "user", "content": f"{existing_tracks_context}How well does the song '{song_title}' by {artist_name} match this prompt: '{prompt}' and fit with the existing songs?"}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            gpt_score = float(response.choices[0].message.content.strip())
            logger.info(f"GPT score for {song_title} - {artist_name}: {gpt_score:.2f}")
            
            # Get artist genres
            try:
                artist_id = track_info['artists'][0]['id']
                artist_info = sp.artist(artist_id)
                genres = artist_info.get('genres', [])
            except Exception as e:
                logger.warning(f"Could not get artist info for track {track_info['id']}: {str(e)}")
                genres = []
            
            # Start with GPT score as base
            score = gpt_score
            
            # Add genre bonus if applicable
            if genres:
                prompt_lower = prompt.lower()
                prompt_genres = []
                for genre_keyword in ['rock', 'pop', 'jazz', 'classical', 'hip hop', 'rap', 'electronic', 'folk', 'country', 'metal', 'indie']:
                    if genre_keyword in prompt_lower:
                        prompt_genres.append(genre_keyword)
                
                # If genres were mentioned and we found matching genres, add bonus
                if prompt_genres and any(g for g in genres if any(pg in g for pg in prompt_genres)):
                    score = min(1.0, score + 0.1)  # Small bonus for genre match
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Error validating track after {max_retries} attempts: {str(e)}")
                return 0.5  # Return neutral score on error after all retries
    
    return 0.5  # Return neutral score if all retries failed

def validate_track_batch(tracks_batch: List[Tuple[dict, str]], sp: spotipy.Spotify) -> List[Tuple[str, float]]:
    """
    Validate a batch of tracks in parallel.
    Returns a list of (track_uri, score) tuples.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Increase workers for large batches
        futures = []
        for track, prompt in tracks_batch:
            futures.append(
                executor.submit(
                    validate_track_for_prompt,
                    track,
                    prompt,
                    sp
                )
            )
        
        for future, (track, _) in zip(futures, tracks_batch):
            try:
                score = future.result(timeout=60)  # Add timeout to prevent hanging
                results.append((track['uri'], score))
            except Exception as e:
                logger.error(f"Error validating track {track.get('name', 'unknown')}: {str(e)}")
                results.append((track['uri'], 0.5))  # Default score on error
    
    return results

def search_spotify_tracks(sp: spotipy.Spotify, suggestions: str, prompt: str = None, validate: bool = False, desired_count: int = 20, existing_tracks: list = None) -> list:
    """Search for tracks on Spotify and return their URIs. Optionally validate matches against prompt."""
    track_uris = []
    rejected_count = 0
    validation_scores = {}
    tracks_to_validate = []
    
    while len(track_uris) < desired_count:
        # Process each line in the suggestions
        for line in suggestions.split('\n'):
            if len(track_uris) >= desired_count and desired_count > 0:  # Only break if desired_count > 0
                break
                
            if '-' in line:
                # Extract title and artist from the suggestion
                title, artist = line.split('-', 1)
                title = title.strip()
                artist = artist.strip()
                
                try:
                    # Search for the track on Spotify
                    results = sp.search(f"{title} {artist}", limit=1, type='track')
                    
                    if results['tracks']['items']:
                        track = results['tracks']['items'][0]
                        track_uri = track['uri']
                        
                        # Skip if we already have this track
                        if track_uri in track_uris or track_uri in [t[0]['uri'] for t in tracks_to_validate]:
                            continue
                        
                        # If validation is enabled and we have a prompt, add to validation batch
                        if validate and prompt:
                            tracks_to_validate.append((track, prompt))
                        else:
                            # No validation, just add the track
                            track_uris.append(track_uri)
                            logger.info(f"Found track: {title} - {artist}")
                except Exception as e:
                    logger.error(f"Error searching for track {title}: {str(e)}")
        
        # If we have tracks to validate, process them in parallel batches
        if tracks_to_validate:
            BATCH_SIZE = 20  # Increase batch size for better performance
            for i in range(0, len(tracks_to_validate), BATCH_SIZE):
                batch = tracks_to_validate[i:i + BATCH_SIZE]
                batch_results = validate_track_batch(batch, sp)
                
                for uri, score in batch_results:
                    validation_scores[uri] = score
                    if score >= 0.5:
                        track_uris.append(uri)
                        track = next(t[0] for t in batch if t[0]['uri'] == uri)
                        logger.info(f"Validated and added track: {track['name']} (score: {score:.2f})")
                    else:
                        rejected_count += 1
                        track = next(t[0] for t in batch if t[0]['uri'] == uri)
                        logger.info(f"Rejected track: {track['name']} (low score: {score:.2f})")
            
            tracks_to_validate = []  # Clear the batch
        
        # If we still need more tracks and have a limit, generate more suggestions
        if (desired_count > 0 and len(track_uris) < desired_count):
            logger.info(f"Need {desired_count - len(track_uris)} more tracks, generating additional suggestions...")
            new_suggestions = generate_playlist_suggestions(prompt, max(20, desired_count - len(track_uris)), existing_tracks)
            if new_suggestions:
                suggestions = new_suggestions
            else:
                break  # Break if we can't generate more suggestions
    
    if validate and prompt:
        logger.info(f"Validation results: {len(track_uris)} tracks accepted, {rejected_count} tracks rejected")
    
    return track_uris

@app.route('/')
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Musyka API is running',
        'version': '1.0.0'
    })

def generate_playlist_name(prompt: str) -> str:
    """Generate a creative playlist name based on the prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": """You are a creative playlist naming expert. Your task is to create a short, catchy name (2-4 words) that captures the mood or theme.

RULES:
1. NEVER use any of these words or their variations: AI, Generated, Bot, Artificial, Machine, Automated
2. NEVER include dates, times, or timestamps
3. NEVER use quotes or explanations
4. ONLY return the name itself

Examples of good names:
- For relaxing jazz: Midnight Jazz Lounge
- For workout music: Power Hour Mix
- For summer vibes: Beach Party Grooves
- For classical music: Piano Dreams Collection
- For rock music: Guitar Heroes Anthology
- For meditation: Peaceful Mind Journey

BAD examples (never use these formats):
- AI Generated Summer Mix
- Generated Playlist 2024
- Bot's Music Collection
- [timestamp] Dance Mix
- "Happy Vibes" (with quotes)
- Summer Mix 05-13

Remember: Return ONLY the name, nothing else."""},
                {"role": "user", "content": f"Create a playlist name for: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=30
        )
        
        playlist_name = response.choices[0].message.content.strip().strip('"').strip("'")
        
        # Comprehensive validation
        forbidden_terms = [
            'ai', 'generated', 'bot', 'artificial', 'machine', 'automated',
            'playlist', 'mix', 'collection', 'compilation', # Only allow these at the end
            '[', ']', '"', "'", # No special characters
        ]
        
        # Check for timestamps and dates
        has_timestamp = any(char.isdigit() for char in playlist_name)
        
        # Check for forbidden terms not at the end
        name_lower = playlist_name.lower()
        has_forbidden = any(
            term in name_lower.replace(' mix', '').replace(' playlist', '').replace(' collection', '')
            for term in forbidden_terms
        )
        
        if has_timestamp or has_forbidden:
            # Create a simple but natural name from the prompt
            words = [word for word in prompt.split() if not any(term in word.lower() for term in forbidden_terms)][:2]
            if not words:
                words = ['My', 'Music']
            playlist_name = " ".join(word.capitalize() for word in words)
            # Add a natural suffix if it doesn't end with one
            if not any(playlist_name.lower().endswith(suffix) for suffix in [' mix', ' playlist', ' collection']):
                playlist_name += ' Mix'
        
        logger.info(f"Generated playlist name: {playlist_name}")
        return playlist_name
    except Exception as e:
        logger.error(f"Error generating playlist name: {str(e)}")
        # Create a simple but natural name from the prompt
        words = [word for word in prompt.split() if not any(term in word.lower() for term in forbidden_terms)][:2]
        if not words:
            words = ['My', 'Music']
        return " ".join(word.capitalize() for word in words) + ' Mix'

@app.route('/api/generate-playlist', methods=['POST'])
def generate_playlist():
    """Generate a playlist based on the user's prompt."""
    try:
        data = request.json
        prompt = data.get('prompt')
        token_info = data.get('token_info')
        song_count = data.get('song_count', 20)
        validate_songs = data.get('validate_songs', False)
        
        if not prompt or not token_info:
            return jsonify({'error': 'Prompt and token_info are required'}), 400

        # Initialize Spotify client
        sp = get_spotify_client(token_info)
        
        # Get user profile
        user = sp.current_user()
        
        # Generate playlist suggestions
        suggestions = generate_playlist_suggestions(prompt, song_count)
        if not suggestions:
            return jsonify({'error': 'Failed to generate suggestions'}), 500
        
        # Search for tracks on Spotify with optional validation
        track_uris = search_spotify_tracks(
            sp, 
            suggestions, 
            prompt=prompt if validate_songs else None, 
            validate=validate_songs,
            desired_count=song_count
        )
        
        if not track_uris:
            return jsonify({'error': 'No matching tracks found'}), 404
        
        # Generate creative playlist name
        playlist_name = generate_playlist_name(prompt)
        logger.info(f"Creating playlist with name: {playlist_name}")
        
        # Create new playlist
        playlist = sp.user_playlist_create(
            user=user['id'],
            name=playlist_name,
            description=f"Created by Musyka based on: {prompt}"
        )
        
        # Add tracks to playlist
        sp.playlist_add_items(playlist['id'], track_uris)
        
        logger.info(f"Created playlist: {playlist_name} with {len(track_uris)} tracks")
        
        return jsonify({
            'success': True,
            'playlist_url': playlist['external_urls']['spotify'],
            'playlist_id': playlist['id'],
            'playlist_name': playlist_name,
            'suggestions': suggestions,
            'tracks_count': len(track_uris),
            'validation_enabled': validate_songs
        })

    except Exception as e:
        logger.error(f"Error generating playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/spotify', methods=['GET'])
def spotify_auth():
    """Get Spotify authorization URL."""
    try:
        show_dialog = request.args.get('show_dialog', 'false').lower() == 'true'
        auth_url = sp_oauth.get_authorize_url()
        # Add show_dialog parameter to the URL if needed
        if show_dialog:
            auth_url += '&show_dialog=true'
        return jsonify({'auth_url': auth_url})
    except Exception as e:
        logger.error(f"Error getting auth URL: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/spotify/callback', methods=['GET'])
def spotify_callback():
    """Handle Spotify OAuth callback."""
    try:
        code = request.args.get('code')
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Use get_cached_token instead of get_access_token
        token_info = sp_oauth.get_cached_token()
        if not token_info:
            token_info = sp_oauth.get_access_token(code, as_dict=True, check_cache=False)
        
        return jsonify({
            'success': True,
            'token_info': token_info
        })
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists', methods=['GET'])
def get_user_playlists():
    """Get user's playlists."""
    try:
        token_info_str = request.args.get('token_info')
        if not token_info_str or token_info_str == 'null':
            return jsonify({'error': 'token_info is required'}), 400
            
        # Parse the JSON string
        try:
            token_info = json.loads(token_info_str)
            if not token_info or not isinstance(token_info, dict) or 'access_token' not in token_info:
                return jsonify({'error': 'Invalid token_info format'}), 400
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid token_info JSON'}), 400
        
        sp = get_spotify_client(token_info)
        user = sp.current_user()
        
        # Get user's playlists
        playlists = []
        results = sp.current_user_playlists(limit=50)
        while results:
            for playlist in results['items']:
                playlists.append({
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist.get('description', ''),
                    'tracks_count': playlist['tracks']['total'],
                    'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                    'owner': playlist['owner']['display_name'],
                    'is_owner': playlist['owner']['id'] == user['id'],
                    'url': playlist['external_urls']['spotify']
                })
            
            if results['next']:
                results = sp.next(results)
            else:
                break

        return jsonify({
            'success': True,
            'playlists': playlists
        })

    except Exception as e:
        logger.error(f"Error getting playlists: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>', methods=['GET'])
def get_playlist(playlist_id):
    """Get playlist details and tracks."""
    try:
        token_info_str = request.args.get('token_info')
        if not token_info_str or token_info_str == 'null':
            return jsonify({'error': 'token_info is required'}), 400
            
        # Parse the JSON string
        try:
            token_info = json.loads(token_info_str)
            if not token_info or not isinstance(token_info, dict) or 'access_token' not in token_info:
                return jsonify({'error': 'Invalid token_info format'}), 400
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid token_info JSON'}), 400

        sp = get_spotify_client(token_info)
        
        # Get playlist details
        playlist = sp.playlist(playlist_id)
        
        # Get playlist tracks
        tracks = []
        results = sp.playlist_tracks(playlist_id)
        while results:
            for item in results['items']:
                track = item['track']
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'url': track['external_urls']['spotify'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                })
            
            if results['next']:
                results = sp.next(results)
            else:
                break

        return jsonify({
            'success': True,
            'playlist': {
                'id': playlist['id'],
                'name': playlist['name'],
                'description': playlist.get('description', ''),
                'tracks_count': playlist['tracks']['total'],
                'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                'owner': playlist['owner']['display_name'],
                'is_owner': playlist['owner']['id'] == sp.current_user()['id'],
                'url': playlist['external_urls']['spotify']
            },
            'tracks': tracks
        })

    except Exception as e:
        logger.error(f"Error getting playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>/tracks', methods=['POST'])
def add_tracks_to_playlist(playlist_id):
    """Add tracks to an existing playlist."""
    try:
        data = request.json
        token_info = data.get('token_info')
        track_uris = data.get('track_uris', [])
        
        if not token_info or not track_uris:
            return jsonify({'error': 'token_info and track_uris are required'}), 400

        sp = get_spotify_client(token_info)
        
        # Verify playlist exists and user has access
        playlist = sp.playlist(playlist_id)
        if playlist['owner']['id'] != sp.current_user()['id']:
            return jsonify({'error': 'You do not have permission to modify this playlist'}), 403
        
        # Add tracks to playlist
        sp.playlist_add_items(playlist_id, track_uris)
        
        return jsonify({
            'success': True,
            'message': f'Added {len(track_uris)} tracks to playlist'
        })

    except Exception as e:
        logger.error(f"Error adding tracks to playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>/tracks', methods=['DELETE'])
def remove_tracks_from_playlist(playlist_id):
    """Remove tracks from a playlist."""
    try:
        data = request.json
        token_info = data.get('token_info')
        track_uris = data.get('track_uris', [])
        
        if not token_info or not track_uris:
            return jsonify({'error': 'token_info and track_uris are required'}), 400

        sp = get_spotify_client(token_info)
        
        # Verify playlist exists and user has access
        playlist = sp.playlist(playlist_id)
        if playlist['owner']['id'] != sp.current_user()['id']:
            return jsonify({'error': 'You do not have permission to modify this playlist'}), 403
        
        # Remove tracks from playlist
        sp.playlist_remove_all_occurrences_of_items(playlist_id, track_uris)
        
        return jsonify({
            'success': True,
            'message': f'Removed {len(track_uris)} tracks from playlist'
        })

    except Exception as e:
        logger.error(f"Error removing tracks from playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/modify-playlist', methods=['POST'])
def modify_playlist():
    """Add AI-generated songs to an existing playlist."""
    try:
        data = request.json
        playlist_id = data.get('playlist_id')
        token_info = data.get('token_info')
        song_count = data.get('song_count', 20)
        custom_prompt = data.get('prompt', '')
        validate_songs = data.get('validate_songs', False)
        
        if not playlist_id or not token_info:
            return jsonify({'error': 'playlist_id and token_info are required'}), 400

        # Initialize Spotify client
        sp = get_spotify_client(token_info)
        
        # Get existing playlist tracks
        try:
            playlist = sp.playlist(playlist_id)
            existing_tracks = []
            results = sp.playlist_tracks(playlist_id)
            while results:
                for item in results['items']:
                    if item['track']:
                        existing_tracks.append(item['track'])
                if results['next']:
                    results = sp.next(results)
                else:
                    break
            
            user = sp.current_user()
            if playlist['owner']['id'] != user['id']:
                return jsonify({'error': 'You do not have permission to modify this playlist'}), 403
        except Exception as e:
            return jsonify({'error': f'Invalid playlist ID or access denied: {str(e)}'}), 400
        
        # Generate playlist suggestions using the custom prompt or playlist name
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"Songs similar to those in the playlist: {playlist['name']}"
            
        suggestions = generate_playlist_suggestions(prompt, song_count, existing_tracks)
        if not suggestions:
            return jsonify({'error': 'Failed to generate suggestions'}), 500
        
        # Search for tracks on Spotify with optional validation
        track_uris = search_spotify_tracks(
            sp, 
            suggestions, 
            prompt=prompt if validate_songs else None, 
            validate=validate_songs,
            desired_count=song_count,
            existing_tracks=existing_tracks
        )
        
        if not track_uris:
            return jsonify({'error': 'No matching tracks found'}), 404
        
        # Add tracks to playlist in batches of 100
        tracks_added = 0
        batch_size = 100
        
        for i in range(0, len(track_uris), batch_size):
            batch = track_uris[i:i + batch_size]
            sp.playlist_add_items(playlist['id'], batch)
            tracks_added += len(batch)
            logger.info(f"Added batch of {len(batch)} tracks to playlist")
        
        logger.info(f"Modified playlist: {playlist['name']} with {tracks_added} new tracks")
        
        return jsonify({
            'success': True,
            'playlist_url': playlist['external_urls']['spotify'],
            'playlist_id': playlist['id'],
            'playlist_name': playlist['name'],
            'tracks_count': tracks_added,
            'validation_enabled': validate_songs
        })

    except Exception as e:
        logger.error(f"Error modifying playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-track', methods=['POST'])
def search_track():
    """Search for a specific track on Spotify."""
    try:
        data = request.json
        token_info = data.get('token_info')
        query = data.get('query')
        
        if not token_info or not query:
            return jsonify({'error': 'token_info and query are required'}), 400

        sp = get_spotify_client(token_info)
        
        # Search for track
        results = sp.search(query, limit=1, type='track')
        
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return jsonify({
                'success': True,
                'track_uri': track['uri'],
                'track': {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'url': track['external_urls']['spotify'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
            })
        else:
            return jsonify({'error': 'No matching track found'}), 404

    except Exception as e:
        logger.error(f"Error searching for track: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    """Get user's Spotify profile."""
    token_info_str = request.args.get('token_info')
    if not token_info_str:
        return jsonify({'success': False, 'error': 'token_info is required'}), 400

    try:
        token_info = json.loads(token_info_str)
        sp = spotipy.Spotify(auth=token_info['access_token'])
        user = sp.current_user()
        
        return jsonify({
            'success': True,
            'profile': {
                'display_name': user.get('display_name', ''),
                'images': user.get('images', []),
                'followers': user.get('followers', {'total': 0}),
                'product': user.get('product', '')
            }
        })
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tracks/recent', methods=['GET'])
def get_recent_tracks():
    """Get user's recently played tracks."""
    token_info_str = request.args.get('token_info')
    if not token_info_str:
        return jsonify({'success': False, 'error': 'token_info is required'}), 400

    try:
        token_info = json.loads(token_info_str)
        sp = spotipy.Spotify(auth=token_info['access_token'])
        results = sp.current_user_recently_played(limit=5)
        
        recent_tracks = []
        for item in results['items']:
            track = item['track']
            if track and track['id'] not in [t['id'] for t in recent_tracks]:
                recent_tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'url': track['external_urls']['spotify'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                })
        
        return jsonify({
            'success': True,
            'tracks': recent_tracks
        })
    except Exception as e:
        logger.error(f"Error getting recent tracks: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug) 