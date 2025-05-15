#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if ngrok is installed
if ! command_exists ngrok; then
    echo -e "${RED}Error: ngrok is not installed. Please install it first:${NC}"
    echo "npm install -g ngrok"
    exit 1
fi

# Check if backend server is running
if ! curl -s http://localhost:5000 >/dev/null; then
    echo -e "${YELLOW}Warning: Backend server doesn't seem to be running on port 5000${NC}"
    echo "Please start your backend server first"
    exit 1
fi

echo -e "${GREEN}Starting ngrok tunnel...${NC}"

# Kill any existing ngrok processes
pkill ngrok 2>/dev/null || true

# Create a temporary log file
TEMP_LOG=$(mktemp)

# Start ngrok for backend (port 5000)
echo -e "${YELLOW}Initializing ngrok...${NC}"
ngrok http 5000 --log=stdout > "$TEMP_LOG" 2>&1 &
BACKEND_PID=$!

# Wait for ngrok to start and get URL
echo -e "${YELLOW}Waiting for ngrok to initialize...${NC}"
for i in {1..30}; do
    if ! ps -p $BACKEND_PID > /dev/null; then
        echo -e "${RED}Error: ngrok process died${NC}"
        echo "Last few lines of ngrok log:"
        tail -n 20 "$TEMP_LOG"
        rm "$TEMP_LOG"
        exit 1
    fi
    
    # Look for both .ngrok.io and .ngrok.app domains
    BACKEND_URL=$(grep -o 'https://.*\.ngrok\.\(io\|app\)' "$TEMP_LOG" 2>/dev/null | head -n 1)
    if [ ! -z "$BACKEND_URL" ]; then
        break
    fi
    sleep 1
    echo -n "."
done
echo

if [ -z "$BACKEND_URL" ]; then
    echo -e "${RED}Error: Failed to get ngrok URL after 30 seconds${NC}"
    echo "Checking ngrok logs:"
    cat "$TEMP_LOG"
    rm "$TEMP_LOG"
    pkill ngrok
    exit 1
fi

# Clean up temp log
rm "$TEMP_LOG"

echo -e "${GREEN}Backend URL: ${NC}$BACKEND_URL"

# Function to update a specific line in .env file
update_env_line() {
    local file=$1
    local key=$2
    local value=$3
    
    if [ -f "$file" ]; then
        # Check if the key exists in the file
        if grep -q "^$key=" "$file"; then
            # Update the existing line
            sed -i "s|^$key=.*|$key=$value|" "$file"
            echo -e "${GREEN}Updated $key in $file${NC}"
        else
            # Add the new line at the end
            echo "$key=$value" >> "$file"
            echo -e "${GREEN}Added $key to $file${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: $file not found${NC}"
    fi
}

# Update mobile/.env
update_env_line "mobile/.env" "API_URL" "$BACKEND_URL"

# Update backend/.env
update_env_line "backend/.env" "SPOTIFY_REDIRECT_URI" "$BACKEND_URL/api/auth/spotify/callback"

echo -e "\n${GREEN}Tunnel is running!${NC}"
echo -e "${YELLOW}Important:${NC}"
echo "1. Update your Spotify Developer Dashboard with the new redirect URI:"
echo "   $BACKEND_URL/api/auth/spotify/callback"
echo "2. Press Ctrl+C to stop the tunnel"

# Keep the script running and handle cleanup on exit
trap 'echo -e "\n${YELLOW}Stopping ngrok...${NC}"; pkill ngrok; exit 0' INT TERM
wait $BACKEND_PID 