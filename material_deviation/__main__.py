"""Start the standalone PyQt5 material placement deviation UI."""

import argparse
import sys
import traceback

from PyQt5.QtWidgets import QApplication

from .settings import load_settings
from .window import MaterialDeviationWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="RGB material placement deviation UI")
    parser.add_argument("--config", default="configs/config.example.json")
    args = parser.parse_args()
    settings = load_settings(args.config)
    settings.templates_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    weights = str(settings.weights_file) if settings.weights_file.is_file() else None
    if weights is None:
        print(f"SuperPoint weights unavailable: {settings.weights_file}; using ORB fallback")

    def except_hook(exc_type, exception, tb):
        traceback.print_exception(exc_type, exception, tb)

    sys.excepthook = except_hook
    app = QApplication(sys.argv)
    window = MaterialDeviationWindow(settings.camera, weights, settings.cuda, settings)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
