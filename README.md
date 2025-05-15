# Musyka - AI-Powered Spotify Playlist Generator

Musyka is an intelligent playlist generator that uses AI to create personalized Spotify playlists based on your preferences and mood.

## Features

- AI-powered playlist generation using OpenAI
- Seamless Spotify integration
- Modern mobile interface built with Expo/React Native
- Smart playlist suggestions based on your music taste
- Secure token management
- Environment-based configuration

## Prerequisites

- Spotify Premium account
- OpenAI API key
- Node.js and npm
- Python 3.8+

## Setup

### 1. Spotify Developer Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. Add `musyka://auth` to Redirect URIs
4. Note down your Client ID and Client Secret

### 2. Environment Setup

#### Backend (.env)
```bash
cd backend
cp .env.example .env
```
Edit `.env` with your credentials:
```
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
OPENAI_API_KEY=your_openai_api_key
JWT_SECRET=your_random_secret_key
PORT=5000
HOST=0.0.0.0
DEBUG=True
CORS_ORIGINS=http://localhost:3000,exp://localhost:19000
```

#### Mobile (.env)
```bash
cd mobile
cp .env.example .env
```
Edit `.env` with your credentials:
```
API_URL=http://localhost:5000
SPOTIFY_CLIENT_ID=your_spotify_client_id
OPENAI_API_KEY=your_openai_api_key
APP_ENV=development
```

### 3. Backend Setup

1. Create and activate virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   python app.py
   ```

### 4. Mobile App Setup

1. Install dependencies:
   ```bash
   cd mobile
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Run on your device:
   - Install Expo Go on your phone
   - Scan the QR code with:
     - iOS: Use the Camera app
     - Android: Use the Expo Go app

## Development

### Backend
- Uses Flask for the API server
- Implements Spotify OAuth flow
- Integrates with OpenAI API
- Includes logging and error handling
- Type hints and documentation

### Mobile App
- Built with Expo/React Native
- TypeScript for type safety
- Environment-based configuration
- Secure token storage
- Modern UI with React Native Paper

### Using ngrok for Development

To test the app outside your local network, you can use ngrok to create secure tunnels to your local development servers. We've included a script to make this process easier.

1. Install ngrok globally:
   ```bash
   npm install -g ngrok
   ```

2. Run the tunnel script:
   ```bash
   ./tunnel.sh
   ```

The script will:
- Start an ngrok tunnel for the backend server
- Automatically update the necessary environment variables
- Display the ngrok URL for the backend

Note: The ngrok URL will change each time you restart the tunnel. Make sure to update your Spotify Developer Dashboard with the new redirect URI if needed.

## Security Notes

- Never commit `.env` files
- Keep your API keys secure
- Use environment variables for all sensitive data
- JWT tokens are used for session management
- CORS is configured for security

## Troubleshooting

1. **Backend Connection Issues**
   - Ensure the backend is running
   - Check if the port is available
   - Verify CORS settings

2. **Spotify Authentication**
   - Verify redirect URI in Spotify Dashboard
   - Check if Spotify credentials are correct
   - Ensure proper scopes are requested

3. **Mobile App Issues**
   - Clear app cache
   - Reinstall dependencies
   - Check environment variables

## License

MIT 