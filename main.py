import sys
import traceback
from app import App

def main():
    try:
        app = App()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()