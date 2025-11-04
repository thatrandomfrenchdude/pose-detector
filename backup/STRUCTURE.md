# Project Structure

```
pose-detection/
├── main.py                 # Main application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── test_setup.py          # Setup verification script
├── create_sample.py       # Sample image generator
├── sample_person.jpg      # Sample test image
├── README.md              # Comprehensive documentation
└── .venv/                 # Virtual environment (auto-generated)
    └── ...
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test setup:**
   ```bash
   python test_setup.py
   ```

3. **Run real-time detection:**
   ```bash
   python main.py
   ```

4. **Process sample image:**
   ```bash
   python main.py --image sample_person.jpg --output result.jpg
   ```

## Next Development Steps

1. **Add multiple person detection**
2. **Implement pose classification**
3. **Add video file processing**
4. **Create web interface**
5. **Add pose analytics and metrics**