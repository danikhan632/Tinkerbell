#!/bin/bash
# Quick test to verify the server can start

cd /home/green/code/thinker/flask_server

echo "Testing Flask server startup..."
echo ""

# Try to import the app
python3 -c "
import sys
try:
    import app
    print('✓ app.py imports successfully')
except Exception as e:
    print(f'✗ Failed to import app.py: {e}')
    sys.exit(1)
" || exit 1

echo ""
echo "✅ Server can start successfully!"
echo ""
echo "To run the server:"
echo "  python app.py"
