import sys
import pyperclip
import torch
from PIL import Image, ImageGrab
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor


class ScreenshotToLatexApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Load model, tokenizer, and feature extractor
        self.model = VisionEncoderDecoderModel.from_pretrained("Matthijs0/im2latex_base")
        self.tokenizer = AutoTokenizer.from_pretrained("Matthijs0/im2latex_base")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

    def init_ui(self):
        self.setWindowTitle("Screenshot to LaTeX Formula")
        self.setGeometry(100, 100, 600, 400)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Screenshot button
        self.screenshot_button = QPushButton("Take Screenshot", self)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        layout.addWidget(self.screenshot_button)

        # Process button
        self.process_button = QPushButton("Process Screenshot to LaTeX", self)
        self.process_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_button)

        # Display area
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        central_widget.setLayout(layout)

    def take_screenshot(self):
        self.hide()  # Hide the GUI during screenshot
        screenshot = ImageGrab.grab()
        self.show()

        # Save screenshot to memory
        screenshot.save("screenshot.png")
        self.image_label.setPixmap(QPixmap("screenshot.png"))

    def process_image(self):
        try:
            image = Image.open("screenshot.png").convert("RGB")

            # Extract pixel values
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            # Generate LaTeX formula
            generated_ids = self.model.generate(pixel_values)
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Copy result to clipboard
            latex_formula = generated_texts[0]
            pyperclip.copy(latex_formula)

            # Display result
            self.image_label.setText(f"LaTeX Formula Copied to Clipboard:\n{latex_formula}")

        except Exception as e:
            self.image_label.setText(f"Error: {str(e)}")


def inference() -> None:
    app = QApplication(sys.argv)
    window = ScreenshotToLatexApp()
    window.show()
    sys.exit(app.exec_())
