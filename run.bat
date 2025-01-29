@echo off
start ngrok http --url=stable-airedale-powerful.ngrok-free.app 3000
start python app.py
