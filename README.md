# 🚀 Project Setup & Run Instructions

This README explains how to set up and run the entire project from scratch.

---

## ✅ 1. Create a Virtual Environment

Create and activate a virtual environment to keep dependencies isolated.

### **Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### **macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## ✅ 2. Install Required Packages

Once the virtual environment is activated, install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## ✅ 3. Run Initial Setup Scripts (Important)

Before running the main application, run these two scripts first:

### **1. Setup the database**
```bash
python scripts/database_setup.py
```

### **2. Ingest required data**
```bash
python scripts/ingest.py
```

These scripts prepare your database and load the required data.

---

## ✅ 4. Running the Project (Two Terminals Needed)

This project uses **two Python scripts** that must run **in parallel**.  
Open **two separate terminal tabs** after the setup scripts have finished.

---

### 🟦 Terminal Tab 1 — Run `main.py`
```bash
python main.py
```

---

### 🟩 Terminal Tab 2 — Run `SpeechToSpeech.py`
```bash
python SpeechToSpeech.py
```

---

## 🌐 5. Open the Web Interface

After **both** scripts are running, open this URL in your browser:

```
http://127.0.0.1:5001
```

This is where you can interact with the project.

---

## 🎉 All Set!

Your project is now running successfully.  
Keep both terminals active and enjoy using the application.
