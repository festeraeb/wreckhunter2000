#!/usr/bin/env python3
"""
Simple HTTP Server for Rosa Nautical Interface
==============================================

Serves the Rosa nautical interface locally for viewing.

Author: GitHub Copilot
Date: October 12, 2025
"""

import http.server
import socketserver
import os
import webbrowser
from threading import Timer

def serve_nautical_interface():
    """Serve the Rosa nautical interface on local HTTP server"""
    
    # Change to outputs directory
    os.chdir('outputs')
    
    PORT = 8080
    
    # Find an available port
    for port in range(8080, 8090):
        try:
            with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
                PORT = port
                break
        except OSError:
            continue
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"🌐 SERVING ROSA NAUTICAL INTERFACE")
    print(f"=" * 33)
    print(f"   URL: http://localhost:{PORT}/rosa_nautical_interface.html")
    print(f"   Files available:")
    print(f"   • rosa_nautical_interface.html - Interactive nautical chart")
    print(f"   • rosa_nautical_analysis.mbtiles - Vector tiles (5,290 tiles)")
    print(f"   • rosa_enc_overlay.json - ENC chart overlay data")
    print(f"   ")
    print(f"   🗺️ Features:")
    print(f"   • ENC chart style visualization")
    print(f"   • Depth contours (30ft, 60ft, 120ft)")
    print(f"   • Navigation aids and lights")
    print(f"   • Commercial shipping lanes")
    print(f"   • Rosa timeline analysis")
    print(f"   • Interactive boat capability validation")
    print(f"   ")
    print(f"   Press Ctrl+C to stop server")
    
    # Auto-open browser after short delay
    def open_browser():
        webbrowser.open(f'http://localhost:{PORT}/rosa_nautical_interface.html')
    
    Timer(1.0, open_browser).start()
    
    # Start server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    serve_nautical_interface()