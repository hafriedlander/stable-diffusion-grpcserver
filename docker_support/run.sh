#!/bin/bash
if [ -z ${SD_CLOUDFLARE} ]; then
    SERVICE="python"
    if pgrep -x "$SERVICE" >/dev/null
    then
        echo "server is running"
    else 
        /bin/micromamba -r env -n sd-grpc-server run python ./server.py  
    fi
else
    SERVICE="python"
    if pgrep -x "$SERVICE" >/dev/null
    then
            echo "server is running"
    else
        /bin/micromamba -r env -n sd-grpc-server run python ./server.py  &   
    fi
    FILE=./cloudflared-linux-amd64
    if [ ! -f "$FILE" ]; then
        wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
        chmod +x cloudflared-linux-amd64
    fi
    ./cloudflared-linux-amd64 tunnel  --url http://localhost:5000
fi
