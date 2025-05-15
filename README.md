# Musyka - AI-Powered Spotify Playlist Generator

Musyka is an AI-powered Spotify playlist generator that helps you create and enhance playlists based on natural language prompts. Using GPT models and the Spotify API, Musyka can:

- Generate new playlists from text descriptions
- Add songs to existing playlists based on your criteria
- Validate songs to ensure they match your musical preferences
- Create cohesive playlists with consistent musical styles

## Features

- **Natural Language Playlist Creation**: Describe the music you want, and let AI find matching songs
- **Smart Song Validation**: AI evaluates how well each song matches your prompt
- **Playlist Enhancement**: Add more songs to your existing playlists based on their current style
- **Context-Aware Recommendations**: Uses your existing playlist songs for more relevant suggestions

## Tech Stack

### Backend
- Flask API
- Spotify Web API (via Spotipy)
- OpenAI GPT Models
- Concurrent processing for performance

### Mobile App
- React Native
- Expo
- TypeScript
- Spotify Authentication

## Setup Instructions

### Backend Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Raincor5/Musyka.git
   cd Musyka/backend
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   OPENAI_API_KEY=your_openai_api_key
   JWT_SECRET=your_jwt_secret
   DEBUG=true
   ```

5. Run the backend
   ```bash
   python app.py
   ```

### Mobile App Setup

1. Navigate to the mobile app directory
   ```bash
   cd ../mobile-new
   ```

2. Install dependencies
   ```bash
   npm install
   ```

3. Create a `.env` file with the following variables:
   ```
   API_URL=http://localhost:5000
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_REDIRECT_URI=musyka://auth
   APP_ENV=development
   DEBUG=true
   ```

4. Start the app
   ```bash
   npx expo start
   ```

## Spotify Developer Setup

1. Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com)
2. Create a new application and add the following redirect URIs:
   - `musyka://auth` (for mobile app)
   - `http://localhost:5000/api/auth/spotify/callback` (for local development)
3. Make sure to configure the proper scopes in your app.py file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 