# Advanced Object Detection with Distance Estimation 🚀

This Streamlit app performs **real-time object detection** and **depth estimation** using **YOLOv11x** and **MiDaS** models. It provides visual bounding boxes with estimated distances and speaks out the closest detected object using text-to-speech (gTTS + pygame).

---

## 📦 Features

- Real-time webcam-based object detection using **YOLOv11x**
- Depth estimation using **MiDaS_small**
- Class label & distance overlay on frame
- Speech output of closest detected object every few seconds
- Clean and interactive **Streamlit UI** with start/stop controls

---

## 🛠️ Requirements

Make sure you have the following installed:

```bash
pip install streamlit opencv-python numpy torch torchvision torchaudio \
gTTS pygame ultralytics
```

---

## 📁 File Structure

```
.
├── app.py                # Main Streamlit app file (your current script)
├── yolo11x.pt            # YOLOv11x model weights
├── coco.names            # COCO class names (1 per line)
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the URL provided by Streamlit in your browser.

---

## 🎯 Controls

- **Start Detection**: Begins real-time detection and depth estimation.
- **Stop Detection**: Ends detection and releases the webcam.

---

## 📌 Notes

- Ensure `yolo11x.pt` and `coco.names` are in the same directory or update their paths in the code.
- Requires a working webcam and stable internet connection (for gTTS).
- Depth estimation uses a linear approximation from MiDaS normalized depth.

---

## 📄 License

This project is for educational purposes only.

---

## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Intel MiDaS](https://github.com/intel-isl/MiDaS)
- [gTTS (Google Text-to-Speech)](https://pypi.org/project/gTTS/)
